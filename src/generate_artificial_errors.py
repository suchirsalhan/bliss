import json
import re
import random
import logging
import glob
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import lemminflect
    LEMMINFLECT_AVAILABLE = True
except ImportError:
    LEMMINFLECT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ErrorGenerator:
    def __init__(self, approach="precise"):
        self.approach = approach
        logger.info(f"Initializing error generator ({approach})")

        if SPACY_AVAILABLE:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = None
            logger.warning("spaCy not available")

        if approach == "high_coverage" and TRANSFORMERS_AVAILABLE:
            logger.info("Loading MLM pipeline...")
            self.mlm_pipeline = pipeline("fill-mask", model="roberta-base", top_k=15)
        else:
            self.mlm_pipeline = None

        self.substitutions = {
            'DET': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'each', 'every'],
            'ADP': ['of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'about', 'through', 'during'],
            'CCONJ': ['and', 'but', 'or', 'so', 'yet'],
            'SCONJ': ['because', 'since', 'if', 'when', 'while', 'although', 'unless', 'that'],
        }

        logger.info("Ready")

    def parse_error_type(self, error_type: str) -> Tuple[str, str, str]:
        parts = error_type.split(':')
        operation = parts[0]
        pos = parts[1] if len(parts) > 1 else ''
        subtype = parts[2] if len(parts) > 2 else ''
        return operation, pos, subtype

    def find_learner_edit(self, good: str, bad: str) -> Tuple[Optional[int], str, str, str]:
        good_words = good.split()
        bad_words = bad.split()

        for i in range(min(len(good_words), len(bad_words))):
            if good_words[i] != bad_words[i]:
                return i, 'R', good_words[i], bad_words[i]

        if len(good_words) > len(bad_words):
            return len(bad_words), 'M', good_words[len(bad_words)], ''
        elif len(bad_words) > len(good_words):
            return len(good_words), 'U', '', bad_words[len(good_words)]

        return None, '', '', ''

    def get_candidates_by_pos(self, sentence: str, target_pos: str) -> List[Tuple[int, str]]:
        if not self.nlp:
            return []

        doc = self.nlp(sentence)
        candidates = []

        for token in doc:
            if token.pos_ == target_pos:
                candidates.append((token.i, token.text))

        return candidates

    def generate_det_error(self, sentence: str, operation: str, learner_pos: Optional[int]) -> List[str]:
        candidates = []
        if not self.nlp:
            return candidates

        doc = self.nlp(sentence)
        words = sentence.split()

        if operation == 'M':  # remove determiner
            det_positions = self.get_candidates_by_pos(sentence, 'DET')
            for token_pos, det_word in det_positions:
                if learner_pos is None or token_pos != learner_pos:
                    if token_pos < len(words):
                        new_words = words[:token_pos] + words[token_pos+1:]
                        candidates.append(' '.join(new_words))

        elif operation == 'U':  # insert determiner
            for chunk in doc.noun_chunks:
                has_det = any(child.pos_ == "DET" for child in chunk.root.children)
                if not has_det and (learner_pos is None or chunk.start != learner_pos):
                    for det in ['the', 'a', 'this']:
                        new_words = words.copy()
                        new_words.insert(chunk.start, det)
                        candidates.append(' '.join(new_words))

        elif operation == 'R':  # replace determiner
            det_positions = self.get_candidates_by_pos(sentence, 'DET')
            for token_pos, det_word in det_positions:
                if learner_pos is None or token_pos != learner_pos:
                    for replacement in self.substitutions['DET']:
                        if replacement.lower() != det_word.lower() and token_pos < len(words):
                            new_words = words.copy()
                            new_words[token_pos] = replacement
                            candidates.append(' '.join(new_words))

        return candidates[:10]

    def generate_prep_error(self, sentence: str, operation: str, learner_pos: Optional[int]) -> List[str]:
        candidates = []
        prep_positions = self.get_candidates_by_pos(sentence, 'ADP')
        words = sentence.split()

        if operation == 'R':  # replace preposition
            for token_pos, prep_word in prep_positions:
                if learner_pos is None or token_pos != learner_pos:
                    for replacement in self.substitutions['ADP']:
                        if replacement != prep_word.lower() and token_pos < len(words):
                            new_words = words.copy()
                            new_words[token_pos] = replacement
                            candidates.append(' '.join(new_words))

        elif operation == 'M':  # remove preposition
            for token_pos, prep_word in prep_positions:
                if learner_pos is None or token_pos != learner_pos:
                    if token_pos < len(words):
                        new_words = words[:token_pos] + words[token_pos+1:]
                        candidates.append(' '.join(new_words))

        return candidates[:8]

    def generate_noun_error(self, sentence: str, operation: str, subtype: str, learner_pos: Optional[int]) -> List[str]:
        candidates = []
        if not self.nlp or not LEMMINFLECT_AVAILABLE:
            return candidates

        if subtype == 'NUM':
            doc = self.nlp(sentence)
            words = sentence.split()

            for token in doc:
                if token.pos_ == "NOUN" and (learner_pos is None or token.i != learner_pos):
                    if token.i < len(words):
                        lemma = token.lemma_

                        if token.tag_ == "NN":  # singular -> plural
                            plurals = lemminflect.getInflection(lemma, tag='NNS')
                            for plural in plurals[:2]:
                                new_words = words.copy()
                                new_words[token.i] = plural
                                candidates.append(' '.join(new_words))

                        elif token.tag_ == "NNS":  # plural -> singular
                            singulars = lemminflect.getInflection(lemma, tag='NN')
                            for singular in singulars[:2]:
                                new_words = words.copy()
                                new_words[token.i] = singular
                                candidates.append(' '.join(new_words))

        return candidates[:6]

    def generate_verb_error(self, sentence: str, operation: str, subtype: str, learner_pos: Optional[int]) -> List[str]:
        candidates = []
        if not self.nlp or not LEMMINFLECT_AVAILABLE:
            return candidates

        doc = self.nlp(sentence)
        words = sentence.split()

        for token in doc:
            if token.pos_ == "VERB" and (learner_pos is None or token.i != learner_pos):
                if token.i < len(words):
                    lemma = token.lemma_
                    forms_to_try = []

                    if subtype == "TENSE":
                        if token.tag_ == "VBD":  # past -> present
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBZ'))
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBP'))
                        elif token.tag_ in ["VBZ", "VBP"]:  # present -> past
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBD'))

                    elif subtype == "FORM":
                        if token.tag_ == "VBG":  # gerund
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VB'))
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBN'))
                        elif token.tag_ == "VB":  # base
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBG'))
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBZ'))

                    elif subtype == "SVA":
                        if token.tag_ == "VBZ":  # 3rd singular -> plural
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBP'))
                        elif token.tag_ == "VBP":  # plural -> 3rd singular
                            forms_to_try.extend(lemminflect.getInflection(lemma, tag='VBZ'))

                    for form in forms_to_try[:3]:
                        if form != token.text:
                            new_words = words.copy()
                            new_words[token.i] = form
                            candidates.append(' '.join(new_words))

        return candidates[:6]

    def generate_mlm_candidates(self, sentence: str, target_pos: int) -> List[str]:
        if not self.mlm_pipeline:
            return []

        words = sentence.split()
        if target_pos >= len(words):
            return []

        masked_words = words.copy()
        masked_words[target_pos] = "<mask>"
        masked_sentence = ' '.join(masked_words)

        try:
            results = self.mlm_pipeline(masked_sentence)
            candidates = []

            for result in results[:5]:
                token = result['token_str'].strip()
                if token != words[target_pos] and token.isalpha():
                    new_words = words.copy()
                    new_words[target_pos] = token
                    candidates.append(' '.join(new_words))

            return candidates
        except:
            return []

    def generate_artificial_error(self, good: str, bad: str, error_type: str) -> Optional[str]:
        # skip problematic error types
        if any(skip in error_type.upper() for skip in ['SPELL', 'ORTH', 'PUNCT', 'MORPH', 'OTHER', 'WO']):
            return None

        good = good.strip()
        bad = bad.strip()

        operation, pos, subtype = self.parse_error_type(error_type)
        learner_pos, _, _, _ = self.find_learner_edit(good, bad)

        candidates = []

        if pos == 'DET':
            candidates = self.generate_det_error(good, operation, learner_pos)
        elif pos == 'PREP':
            candidates = self.generate_prep_error(good, operation, learner_pos)
        elif pos == 'NOUN':
            candidates = self.generate_noun_error(good, operation, subtype, learner_pos)
        elif pos == 'VERB':
            candidates = self.generate_verb_error(good, operation, subtype, learner_pos)
        elif pos == 'CONJ':
            if not self.nlp:
                return None
            doc = self.nlp(good)
            words = good.split()
            for token in doc:
                if token.pos_ in ["CCONJ", "SCONJ"] and (learner_pos is None or token.i != learner_pos):
                    conj_word = token.text.lower()
                    replacements = self.substitutions.get('CCONJ', []) + self.substitutions.get('SCONJ', [])
                    for replacement in replacements[:3]:
                        if replacement != conj_word and token.i < len(words):
                            new_words = words.copy()
                            new_words[token.i] = replacement
                            candidates.append(' '.join(new_words))

        # try MLM for high coverage approach if few candidates
        if self.approach == "high_coverage" and len(candidates) < 3 and self.nlp:
            doc = self.nlp(good)
            for token in doc:
                if (learner_pos is None or token.i != learner_pos) and token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
                    mlm_candidates = self.generate_mlm_candidates(good, token.i)
                    candidates.extend(mlm_candidates[:2])

        valid_candidates = []
        for candidate in candidates:
            if candidate != good and candidate != bad and len(candidate.strip()) > 0:
                length_ratio = len(candidate) / len(good) if len(good) > 0 else 0
                if 0.7 <= length_ratio <= 1.3:
                    valid_candidates.append(candidate)

        return valid_candidates[0] if valid_candidates else None


def process_single_file(input_path: str, output_path: str, approach: str = "precise") -> Dict[str, int]:
    logger.info(f"Processing {input_path} -> {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    generator = ErrorGenerator(approach=approach)

    with open(input_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        pairs = data
        original_structure = "list"
    else:
        pairs = data.get('pairs', [])
        original_structure = "dict"

    processed_pairs = []
    success_count = 0

    for i, item in enumerate(pairs):
        if i % 1000 == 0 and i > 0:
            success_rate = success_count / i * 100
            logger.info(f"  {i}/{len(pairs)} ({success_rate:.1f}%)")

        good = item.get('sentence_good') or item.get('corrected_sentence')
        bad = item.get('sentence_bad') or item.get('original_sentence')
        error_type = item.get('target_edit_type') or item.get('error_type_errant') or item.get('error_type')

        if not good or not bad or not error_type:
            processed_pairs.append(item)
            continue

        artificial = generator.generate_artificial_error(good, bad, error_type)

        if artificial:
            new_item = item.copy()
            new_item['artificial_error'] = artificial
            processed_pairs.append(new_item)
            success_count += 1
        else:
            processed_pairs.append(item)

    if original_structure == "list":
        output_data = processed_pairs
    else:
        output_data = data.copy()
        output_data['pairs'] = processed_pairs
        output_data['artificial_generation_stats'] = {
            'approach': approach,
            'total_pairs': len(pairs),
            'successful_generations': success_count,
            'success_rate': success_count / len(pairs) if pairs else 0
        }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    success_rate = success_count / len(pairs) * 100 if pairs else 0
    logger.info(f"Saved {success_count}/{len(pairs)} artificial errors ({success_rate:.1f}%)")

    return {
        'total_pairs': len(pairs),
        'successful_generations': success_count,
        'success_rate': success_rate
    }


def process_writeandimprove(input_dir: str = "/Users/yuangao/babyLM/data/individual_edits",
                           output_dir: str = "/Users/yuangao/babyLM/data/bliss_wi_coverage",
                           approach: str = "precise"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'dev']:
        input_file = input_path / f"writeandimprove_{split}_individual_edits.json"

        if approach == "precise":
            output_file = output_path / f"writeandimprove_{split}_precise_individual_edits.json"
        else:
            output_file = output_path / f"writeandimprove_{split}_coverage_individual_edits.json"

        if input_file.exists():
            process_single_file(str(input_file), str(output_file), approach)
        else:
            logger.warning(f"Input file not found: {input_file}")


def process_multiple_datasets(efcamdat_input: str = "/Users/yuangao/babyLM/data/efcamdat_bliss",
                             fce_input: str = "/Users/yuangao/babyLM/data/fce_errant_bliss",
                             efcamdat_output: str = "/Users/yuangao/babyLM/data/bliss_efcamdat_coverage",
                             fce_output: str = "/Users/yuangao/babyLM/data/bliss_fce_coverage",
                             approach: str = "precise"):

    logger.info(f"Processing EFCamDat with {approach} approach")
    efcamdat_files = glob.glob(f"{efcamdat_input}/*_single_edit_with_foils.json")
    Path(efcamdat_output).mkdir(parents=True, exist_ok=True)

    for input_file in efcamdat_files:
        if Path(input_file).stat().st_size > 100 * 1024 * 1024:  # skip files > 100MB
            logger.info(f"Skipping large file: {input_file}")
            continue

        filename = Path(input_file).name
        output_file = f"{efcamdat_output}/{filename.replace('_single_edit_with_foils', f'_{approach}_artificial')}"
        process_single_file(input_file, output_file, approach)

    logger.info(f"Processing FCE with {approach} approach")
    fce_files = glob.glob(f"{fce_input}/*_single_edit_with_foils.json")
    Path(fce_output).mkdir(parents=True, exist_ok=True)

    for input_file in fce_files:
        filename = Path(input_file).name
        output_file = f"{fce_output}/{filename.replace('_single_edit_with_foils', f'_{approach}_artificial')}"
        process_single_file(input_file, output_file, approach)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate artificial errors for BLIMP datasets")
    parser.add_argument("--approach", choices=["precise", "high_coverage"], default="precise")
    parser.add_argument("--datasets", choices=["writeandimprove", "efcamdat", "fce", "all"], default="all")
    parser.add_argument("--input-file", help="Single input file")
    parser.add_argument("--output-file", help="Single output file")

    args = parser.parse_args()

    if args.input_file and args.output_file:
        process_single_file(args.input_file, args.output_file, args.approach)
    else:
        if args.datasets in ["writeandimprove", "all"]:
            process_writeandimprove(approach=args.approach)

        if args.datasets in ["efcamdat", "fce", "all"]:
            process_multiple_datasets(approach=args.approach)