import argparse
import json
import math
import sys
import time
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Install with: pip install numpy")

# constants
BPC_FACTOR = 1.0 / math.log(2.0)  # nats -> bits
DEFAULT_TAU = 0.10
TIE_TOL = 1e-6
EPS = 1e-9

# Removed fidelity weights - using simplified metrics


@dataclass
class SentenceScore:
    n_tokens: int
    nll_nats_per_token: float
    nll_bits_per_token: float


@dataclass
class ItemTriplet:
    file: str
    idx: int
    id: Optional[str]
    l1: Optional[str]
    good: str
    bad: str
    art: str
    bpt_good: float
    bpt_bad: float
    bpt_art: float


class ModelEvaluator:
    def __init__(self, model_name_or_path: str, device: Optional[str] = None,
                 load_in_8bit: bool = False, load_in_4bit: bool = False,
                 dtype: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required for evaluation")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        cfg = AutoConfig.from_pretrained(model_name_or_path)
        if getattr(cfg, "is_encoder_decoder", False):
            print("[ERROR] Seq2Seq models are not supported.", file=sys.stderr)
            sys.exit(1)

        # load tokenizer
        def _load_tok(name_or_path: str):
            try:
                return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
            except Exception as e_fast:
                print(f"[WARN] Fast tokenizer failed: {e_fast}")
                print("[INFO] Falling back to slow tokenizer")
                try:
                    return AutoTokenizer.from_pretrained(name_or_path, use_fast=False)
                except Exception as e_slow:
                    raise RuntimeError(f"Tokenizer load failed. Fast={e_fast} Slow={e_slow}")

        self.tok = _load_tok(model_name_or_path)
        if self.tok.pad_token is None:
            if self.tok.eos_token is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.tok.pad_token_id

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, None: None}
        torch_dtype = dtype_map.get(dtype, None)

        mkw = {}
        if load_in_8bit:
            mkw["load_in_8bit"] = True
            mkw["device_map"] = "auto"
        elif load_in_4bit:
            mkw["load_in_4bit"] = True
            mkw["device_map"] = "auto"
        else:
            if torch_dtype is not None:
                mkw["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **mkw)
        if not (load_in_8bit or load_in_4bit):
            self.model.to(self.device)
        if len(self.tok) > self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tok))
        self.model.eval()

        self.cache: Dict[str, SentenceScore] = {}

    @torch.no_grad()
    def score_batch(self, sentences: List[str], max_length: Optional[int] = None) -> List[SentenceScore]:
        results: List[Optional[SentenceScore]] = [None] * len(sentences)
        to_compute = [(i, s) for i, s in enumerate(sentences) if s not in self.cache]

        if to_compute:
            texts = [s for _, s in to_compute]
            enc = self.tok(
                texts, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt", add_special_tokens=True
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            if self.device == "cuda":
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
            elif self.device == "mps":
                input_ids = input_ids.to("mps")
                attention_mask = attention_mask.to("mps")
                labels = labels.to("mps")

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(dtype=torch.long)
            shift_mask = (shift_labels != -100)

            loss_tok = F.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels,
                ignore_index=-100, reduction='none'
            )
            tok_nll = loss_tok * shift_mask
            seq_nll_sum = tok_nll.sum(dim=1)
            seq_tok_count = shift_mask.sum(dim=1).clamp_min(1)

            for (i_local, text), nll_sum, n_tok in zip(to_compute, seq_nll_sum.tolist(), seq_tok_count.tolist()):
                n_tok = int(n_tok)
                nll_nats_per_token = float(nll_sum / max(n_tok, 1))
                sc = SentenceScore(
                    n_tokens=n_tok,
                    nll_nats_per_token=nll_nats_per_token,
                    nll_bits_per_token=nll_nats_per_token * BPC_FACTOR
                )
                self.cache[text] = sc
                results[i_local] = sc

        # fill in cached results
        for i, sentence in enumerate(sentences):
            if results[i] is None:
                results[i] = self.cache[sentence]

        return [r for r in results if r is not None]


def load_dataset_files(input_dir: str) -> List[str]:
    return [
        os.path.join(input_dir, n)
        for n in sorted(os.listdir(input_dir))
        if n.lower().endswith((".json", ".jsonl"))
    ]


def iter_json_records(path: str) -> Iterator[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # fallback: JSON Lines
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    continue
        return

    # structured JSON
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(data, dict):
        pairs = data.get("pairs")
        if isinstance(pairs, list):
            for obj in pairs:
                if isinstance(obj, dict):
                    yield obj


def compute_metrics(items: List[ItemTriplet], tau: float = DEFAULT_TAU) -> Dict[str, Any]:
    if not items:
        return {"error": "No items to evaluate"}

    n = len(items)

    # compute basic metrics
    margins = []
    wins_at_0 = 0
    wins_at_tau = 0
    correct_best = 0
    gap_scores = []

    for item in items:
        margin = item.bpt_art - item.bpt_bad  # margin = delta_r - delta_h
        margins.append(margin)

        # wins
        if abs(margin) <= TIE_TOL:
            wins_at_0 += 0.5  # tie
        elif margin > TIE_TOL:
            wins_at_0 += 1

        if margin >= tau:
            wins_at_tau += 1

        # correct best (sanity check)
        if item.bpt_good < min(item.bpt_bad, item.bpt_art):
            correct_best += 1

        # normalized gap score
        delta_h = item.bpt_bad - item.bpt_good
        delta_r = item.bpt_art - item.bpt_good
        denominator = abs(delta_r) + abs(delta_h) + EPS
        g = margin / denominator
        gap_scores.append(g)

    # headline metrics (0-100 scale)
    rp_at_0 = 100.0 * (wins_at_0 / n)
    rp_at_tau = 100.0 * (wins_at_tau / n)
    ngs = 100.0 * (0.5 + 0.5 * (sum(gap_scores) / n))
    cps = 100.0 * (correct_best / n)

    # compute significance vs random baseline (50)
    significance = {}
    if NUMPY_AVAILABLE:
        # binomial tests for proportion metrics
        from scipy import stats
        try:
            # exact binomial test for RP@0 (exclude ties)
            n_decisive = sum(1 for m in margins if abs(m) > TIE_TOL)
            n_wins = sum(1 for m in margins if m > TIE_TOL)
            if n_decisive > 0:
                p_rp0 = stats.binom_test(n_wins, n_decisive, 0.5, alternative='two-sided')
                significance['RP_at_0'] = {"p_value": p_rp0, "test": "binomial"}

            # binomial test for RP@tau
            p_rptau = stats.binom_test(wins_at_tau, n, 0.5, alternative='two-sided')
            significance['RP_at_tau'] = {"p_value": p_rptau, "test": "binomial"}

            # binomial test for CPS
            p_cps = stats.binom_test(correct_best, n, 0.5, alternative='two-sided')
            significance['CPS'] = {"p_value": p_cps, "test": "binomial"}

            # t-test for NGS (gap scores vs 0)
            if len(gap_scores) > 1:
                t_stat, p_ngs = stats.ttest_1samp(gap_scores, 0.0)
                significance['NGS'] = {"p_value": p_ngs, "test": "t-test"}

        except ImportError:
            significance = {"error": "scipy required for significance tests"}

    return {
        "headline_metrics": {
            "RP_at_0": rp_at_0,
            "RP_at_tau": rp_at_tau,
            "NGS": ngs,
            "CPS": cps
        },
        "diagnostic_metrics": {
            "n_items": n,
            "margin_stats": {
                "mean": sum(margins) / n,
                "median": sorted(margins)[n//2] if n > 0 else 0.0,
                "std": (sum((m - sum(margins)/n)**2 for m in margins) / n)**0.5 if n > 1 else 0.0
            }
        },
        "significance": significance
    }


def evaluate_model_on_dataset(model_path: str, data_dir: str, output_dir: str = None,
                            batch_size: int = 16, tau: float = DEFAULT_TAU,
                            language: str = None, max_length: int = None,
                            device: str = None, load_in_8bit: bool = False,
                            load_in_4bit: bool = False, dtype: str = None) -> Dict[str, Any]:

    print(f"Loading model: {model_path}")
    evaluator = ModelEvaluator(
        model_name_or_path=model_path,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
    )
    print(f"Using device: {evaluator.device}")

    files = load_dataset_files(data_dir)
    if not files:
        print(f"No JSON files found in {data_dir}")
        return {"error": "No data files found"}

    print(f"Found {len(files)} data files")

    # collect all sentences for scoring
    triples: List[Tuple[str, int, Dict[str, Any]]] = []
    sentences: List[str] = []

    for fpath in files:
        for idx, rec in enumerate(iter_json_records(fpath)):
            g = rec.get("sentence_good")
            b = rec.get("sentence_bad")
            a = rec.get("artificial_error")

            if not (isinstance(g, str) and isinstance(b, str) and isinstance(a, str)):
                continue

            # language filtering
            if language:
                l1 = rec.get("l1_language", "")
                if l1.lower() != language.lower():
                    continue

            triples.append((fpath, idx, rec))
            sentences.extend([g, b, a])

    if not triples:
        print("No valid evaluation items found")
        return {"error": "No valid items"}

    print(f"Scoring {len(sentences)} sentences...")

    # score in batches
    for i in range(0, len(sentences), batch_size):
        evaluator.score_batch(sentences[i:i+batch_size], max_length=max_length)
        if (i // batch_size) % 20 == 0:
            print(f"  scored {min(i+batch_size, len(sentences))}/{len(sentences)}")

    # build evaluation items
    items: List[ItemTriplet] = []
    for (fpath, idx, rec) in triples:
        g = rec["sentence_good"]
        b = rec["sentence_bad"]
        a = rec["artificial_error"]
        l1 = rec.get("l1_language")
        rid = rec.get("UID", f"{os.path.basename(fpath)}_{idx}")

        sc_g = evaluator.cache[g]
        sc_b = evaluator.cache[b]
        sc_a = evaluator.cache[a]

        items.append(ItemTriplet(
            file=os.path.basename(fpath), idx=idx, id=rid, l1=l1,
            good=g, bad=b, art=a,
            bpt_good=sc_g.nll_bits_per_token,
            bpt_bad=sc_b.nll_bits_per_token,
            bpt_art=sc_a.nll_bits_per_token
        ))

    print(f"Computing metrics for {len(items)} items...")
    results = compute_metrics(items, tau=tau)

    # add metadata
    results["metadata"] = {
        "model": model_path,
        "data_dir": data_dir,
        "language_filter": language,
        "tau": tau,
        "n_files": len(files),
        "batch_size": batch_size,
        "device": evaluator.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # save summary
        model_name = os.path.basename(model_path).replace("/", "_")
        lang_suffix = f"_{language}" if language else ""
        summary_file = os.path.join(output_dir, f"{model_name}{lang_suffix}_results.json")

        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {summary_file}")

        # save detailed per-item results
        detailed_file = os.path.join(output_dir, f"{model_name}{lang_suffix}_detailed.jsonl")
        with open(detailed_file, 'w') as f:
            for item in items:
                f.write(json.dumps({
                    "id": item.id,
                    "l1_language": item.l1,
                    "sentence_good": item.good,
                    "sentence_bad": item.bad,
                    "artificial_error": item.art,
                    "bpt_good": item.bpt_good,
                    "bpt_bad": item.bpt_bad,
                    "bpt_art": item.bpt_art,
                    "margin": item.bpt_art - item.bpt_bad
                }, ensure_ascii=False) + "\n")
        print(f"Detailed results saved to {detailed_file}")

    return results


def print_results(results: Dict[str, Any]):
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    meta = results["metadata"]
    print(f"Model: {meta['model']}")
    print(f"Language: {meta.get('language_filter', 'all')}")
    print(f"Items: {results['diagnostic_metrics']['n_items']}")

    print(f"\nHEADLINE METRICS (0-100 scale):")
    print("-" * 40)
    headline = results["headline_metrics"]
    for metric, value in headline.items():
        print(f"{metric:12}: {value:6.2f}")

    print(f"\nDIAGNOSTIC METRICS:")
    print("-" * 40)
    diag = results["diagnostic_metrics"]
    margin_stats = diag["margin_stats"]
    print(f"Margin mean:     {margin_stats['mean']:6.4f} bits/token")
    print(f"Margin median:   {margin_stats['median']:6.4f} bits/token")
    print(f"Margin std:      {margin_stats['std']:6.4f} bits/token")

    # significance tests
    if "significance" in results and results["significance"]:
        print(f"\nSIGNIFICANCE vs RANDOM (p-values):")
        sig = results["significance"]
        for metric, test_result in sig.items():
            if isinstance(test_result, dict) and "p_value" in test_result:
                p_val = test_result["p_value"]
                test_name = test_result.get("test", "unknown")
                print(f"  {metric:12}: p = {p_val:.4f} ({test_name})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate language model on BLISS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate GPT-2 on all data
  python evaluate_model.py gpt2 /path/to/bliss/data --output results/

  # Evaluate on Spanish speakers only
  python evaluate_model.py microsoft/DialoGPT-medium /path/to/data --language Spanish

  # Evaluate with custom settings
  python evaluate_model.py /path/to/local/model /path/to/data --batch-size 8 --tau 0.15
        """
    )

    parser.add_argument("model", help="Model path or HuggingFace model name")
    parser.add_argument("data_dir", help="Directory containing BLISS dataset JSON files")
    parser.add_argument("--output", default=None, help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Margin threshold for RP@tau metric")
    parser.add_argument("--language", default=None, help="Filter by L1 language (e.g., 'Spanish')")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--device", default=None, help="Device (cuda, cpu, mps)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit precision")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit precision")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], help="Model dtype")

    args = parser.parse_args()

    try:
        results = evaluate_model_on_dataset(
            model_path=args.model,
            data_dir=args.data_dir,
            output_dir=args.output,
            batch_size=args.batch_size,
            tau=args.tau,
            language=args.language,
            max_length=args.max_length,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            dtype=args.dtype
        )

        print_results(results)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()