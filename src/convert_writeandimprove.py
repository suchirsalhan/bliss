import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class M2Edit:
    start_offset: int
    end_offset: int
    error_type: str
    correction: str
    sentence_index: int


@dataclass
class MergedEdit:
    start_offset: int
    end_offset: int
    error_types: List[str]
    correction: str
    sentence_index: int
    original_edits: List[M2Edit]


def parse_m2_file_aligned(m2_path: str) -> Dict[int, List[M2Edit]]:
    edits_by_sentence = {}
    current_sentence_idx = 0

    with open(m2_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line.startswith('S '):
            edits_by_sentence[current_sentence_idx] = []

            j = i + 1
            while j < len(lines):
                edit_line = lines[j].strip()

                if not edit_line or edit_line.startswith('S '):
                    break

                if edit_line.startswith('A '):
                    edit_line_content = edit_line[2:]
                    edit = parse_m2_edit_aligned(edit_line_content, current_sentence_idx)

                    if edit:
                        edits_by_sentence[current_sentence_idx].append(edit)

                j += 1

            current_sentence_idx += 1
            i = j
        else:
            i += 1

    # filter out noop edits
    filtered_edits = {}
    total_edits = 0
    for sent_idx, edits in edits_by_sentence.items():
        valid_edits = [edit for edit in edits if not is_noop_edit(edit)]
        filtered_edits[sent_idx] = valid_edits
        total_edits += len(valid_edits)

    print(f"Parsed {current_sentence_idx} sentences, found {total_edits} valid edits")
    return filtered_edits


def parse_m2_edit_aligned(edit_line: str, sentence_idx: int) -> Optional[M2Edit]:
    parts = edit_line.split('|||')
    if len(parts) < 3:
        return None

    try:
        offset_parts = parts[0].strip().split()
        if len(offset_parts) != 2:
            return None

        start_offset = int(offset_parts[0])
        end_offset = int(offset_parts[1])
        error_type = parts[1]
        correction = parts[2] if len(parts) > 2 else ""

        return M2Edit(
            start_offset=start_offset,
            end_offset=end_offset,
            error_type=error_type,
            correction=correction,
            sentence_index=sentence_idx
        )
    except (ValueError, IndexError):
        return None


def is_noop_edit(edit: M2Edit) -> bool:
    return (edit.start_offset == -1 and edit.end_offset == -1) or edit.error_type == 'noop'


def merge_continuous_edits(edits: List[M2Edit]) -> List[MergedEdit]:
    if not edits:
        return []

    sorted_edits = sorted(edits, key=lambda e: (e.start_offset, e.end_offset))

    merged = []
    current_group = [sorted_edits[0]]

    for edit in sorted_edits[1:]:
        last_edit = current_group[-1]

        # consecutive insertions at same position
        if (edit.start_offset == last_edit.end_offset and
            edit.start_offset == edit.end_offset and
            last_edit.start_offset == last_edit.end_offset):
            current_group.append(edit)
        # related position changes
        elif (edit.start_offset == last_edit.start_offset and
              edit.end_offset == last_edit.end_offset + 1):
            current_group.append(edit)
        else:
            merged.append(create_merged_edit(current_group))
            current_group = [edit]

    merged.append(create_merged_edit(current_group))
    return merged


def create_merged_edit(edits: List[M2Edit]) -> MergedEdit:
    if len(edits) == 1:
        edit = edits[0]
        return MergedEdit(
            start_offset=edit.start_offset,
            end_offset=edit.end_offset,
            error_types=[edit.error_type],
            correction=edit.correction,
            sentence_index=edit.sentence_index,
            original_edits=edits
        )

    start_offset = min(e.start_offset for e in edits)
    end_offset = max(e.end_offset for e in edits)
    error_types = [e.error_type for e in edits]

    corrections = []
    for edit in sorted(edits, key=lambda e: (e.start_offset, e.end_offset)):
        if edit.correction:
            corrections.append(edit.correction)

    combined_correction = ' '.join(corrections)

    return MergedEdit(
        start_offset=start_offset,
        end_offset=end_offset,
        error_types=error_types,
        correction=combined_correction,
        sentence_index=edits[0].sentence_index,
        original_edits=edits
    )


def apply_merged_edit_to_tokens(tokens: List[str], merged_edit: MergedEdit) -> List[str]:
    result_tokens = tokens[:]

    for edit in reversed(sorted(merged_edit.original_edits, key=lambda e: e.start_offset)):
        result_tokens = apply_m2_edit_to_tokens(result_tokens, edit)

    return result_tokens


def apply_m2_edit_to_tokens(tokens: List[str], edit: M2Edit) -> List[str]:
    result_tokens = tokens[:]

    if edit.start_offset >= 0 and edit.end_offset >= 0:
        if edit.start_offset == edit.end_offset:
            # insertion
            correction_tokens = edit.correction.split() if edit.correction else []
            if edit.start_offset <= len(result_tokens):
                result_tokens[edit.start_offset:edit.start_offset] = correction_tokens
        else:
            # replacement or deletion
            correction_tokens = edit.correction.split() if edit.correction else []
            if edit.start_offset <= len(result_tokens) and edit.end_offset <= len(result_tokens):
                result_tokens[edit.start_offset:edit.end_offset] = correction_tokens

    return result_tokens


def create_individual_minimal_pairs(orig_sent: str, m2_edits: List[M2Edit], split: str,
                                   base_pair_id: int, sentence_idx: int, essay_id: str,
                                   essay_meta: Dict) -> List[Dict]:
    if not m2_edits:
        return []

    merged_edits = merge_continuous_edits(m2_edits)

    pairs = []
    pair_id = base_pair_id

    for merged_edit in merged_edits:
        sentence_bad = orig_sent
        corrected_tokens = apply_merged_edit_to_tokens(orig_sent.split(), merged_edit)
        sentence_good = ' '.join(corrected_tokens)

        if not is_good_minimal_pair(sentence_bad, sentence_good):
            continue

        error_type_str = '+'.join(merged_edit.error_types)

        blimp_entry = create_blimp_entry(
            sentence_bad, sentence_good, error_type_str,
            split, pair_id, sentence_idx, essay_id, essay_meta,
            merged_edit, orig_sent, sentence_good,
            version="individual"
        )

        pairs.append(blimp_entry)
        pair_id += 1

    return pairs



def create_blimp_entry(sentence_bad: str, sentence_good: str, error_type: str,
                      split: str, pair_id: int, sentence_idx: int, essay_id: str,
                      essay_meta: Dict, merged_edit: Optional[MergedEdit],
                      orig_sent: str, corr_sent: str, version: str) -> Dict:

    blimp_entry = {
        "sentence_good": sentence_good,
        "sentence_bad": sentence_bad,
        "UID": f"writeandimprove_{version}_{split}_{pair_id}",

        "dataset": "writeandimprove_2024",
        "split": split,
        "version": version,
        "error_type_errant": error_type,
        "sentence_id": f"{essay_id}_sentence_{sentence_idx}",
        "essay_id": essay_id,

        "l1_language": essay_meta.get('L1', 'unknown'),
        "auto_cefr": essay_meta.get('auto_cefr', 'unknown'),
        "human_cefr": essay_meta.get('human_cefr', 'unknown'),
        "prompt_id": essay_meta.get('public_prompt_id', 'unknown'),
        "user_id": essay_meta.get('public_user_id', 'unknown'),

        "num_edits": len(merged_edit.original_edits) if merged_edit else 0,
        "edit_positions": f"{merged_edit.start_offset}:{merged_edit.end_offset}" if merged_edit else "unknown"
    }

    return blimp_entry


def is_good_minimal_pair(orig: str, corr: str) -> bool:
    orig_words = orig.split()
    corr_words = corr.split()

    if len(orig_words) < 3 or len(orig_words) > 50:
        return False

    if orig.strip() == corr.strip():
        return False

    word_diff = abs(len(orig_words) - len(corr_words))
    if word_diff > 5:
        return False

    return True


# Removed check_lexical_similarity function - no longer needed


def load_essay_metadata(tsv_path: str) -> Dict[str, Dict]:
    metadata = {}

    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                essay_id = row.get('public_essay_id', '').strip('"')
                clean_row = {}
                for key, value in row.items():
                    clean_row[key] = value.strip('"') if isinstance(value, str) else value
                metadata[essay_id] = clean_row
    except Exception as e:
        print(f"Error loading essay metadata: {e}")

    return metadata


def create_dual_version_minimal_pairs(data_dir: str, split: str, essay_metadata: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
    data_path = Path(data_dir)

    if split == 'test':
        print(f"Skipping {split} split - no corrections available")
        return [], []

    orig_path = data_path / f"en-writeandimprove2024-final-versions-{split}-sentences.orig"
    corr_path = data_path / f"en-writeandimprove2024-final-versions-{split}-sentences.corr"
    ids_path = data_path / f"en-writeandimprove2024-final-versions-{split}-sentences.ids"
    m2_path = data_path / f"en-writeandimprove2024-final-versions-{split}-sentences.m2"

    if not all(p.exists() for p in [orig_path, corr_path]):
        print(f"Missing required files for {split}")
        return [], []

    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_sentences = [line.strip() for line in f]

    with open(corr_path, 'r', encoding='utf-8') as f:
        corr_sentences = [line.strip() for line in f]

    essay_ids = []
    if ids_path.exists():
        with open(ids_path, 'r', encoding='utf-8') as f:
            essay_ids = [line.strip().strip('"') for line in f]

    print(f"Loaded {len(orig_sentences)} original sentences")
    print(f"Loaded {len(corr_sentences)} corrected sentences")

    m2_edits_by_sentence = {}
    if m2_path.exists():
        m2_edits_by_sentence = parse_m2_file_aligned(str(m2_path))
        print(f"Parsed M2 file with {len(m2_edits_by_sentence)} sentence entries")

    individual_pairs = []
    individual_pair_id = 0
    sentences_with_changes = 0

    for i, orig_sent in enumerate(orig_sentences):
        if i >= len(corr_sentences):
            break

        corr_sent = corr_sentences[i]

        if orig_sent.strip() == corr_sent.strip():
            continue

        sentences_with_changes += 1
        essay_id = essay_ids[i] if i < len(essay_ids) else f"essay_{i // 10}"
        essay_meta = essay_metadata.get(essay_id, {})

        m2_edits = m2_edits_by_sentence.get(i, [])

        individual_sentence_pairs = create_individual_minimal_pairs(
            orig_sent, m2_edits, split, individual_pair_id, i, essay_id, essay_meta
        )
        individual_pairs.extend(individual_sentence_pairs)
        individual_pair_id += len(individual_sentence_pairs)

    print(f"Processed {sentences_with_changes} sentences with changes")
    print(f"Created {len(individual_pairs)} individual minimal pairs")

    return individual_pairs


def convert_writeandimprove_dataset(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    individual_output_path = output_path / "individual_edits"

    individual_output_path.mkdir(parents=True, exist_ok=True)

    print(f"Converting Write&Improve data to BLIMP format...")
    print(f"Input directory: {data_path}")
    print(f"Output directory: {individual_output_path}")

    essay_metadata_path = data_path / "en-writeandimprove2024-final-versions-m2-essay-info.tsv"
    essay_metadata = {}

    if essay_metadata_path.exists():
        essay_metadata = load_essay_metadata(str(essay_metadata_path))
        print(f"Loaded metadata for {len(essay_metadata)} essays")
    else:
        print("No essay metadata file found")

    all_individual_data = {}

    for split in ['train', 'dev']:
        print(f"\nProcessing {split} split...")

        individual_pairs = create_dual_version_minimal_pairs(
            data_dir, split, essay_metadata
        )

        if not individual_pairs:
            print(f"No data found for {split} split")
            continue

        all_individual_data[split] = individual_pairs

        individual_file = individual_output_path / f"writeandimprove_{split}_individual_edits.json"
        individual_dataset = {
            "phenomenon": f"writeandimprove_{split}_individual",
            "description": f"Write&Improve 2024 {split} - Individual edit minimal pairs (with continuous edit merging)",
            "version": "individual_edits",
            "pairs": individual_pairs
        }

        with open(individual_file, 'w', encoding='utf-8') as f:
            json.dump(individual_dataset, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(individual_pairs)} individual pairs to {individual_file}")

    if all_individual_data:
        combined_individual = individual_output_path / "writeandimprove_all_individual_edits.json"
        all_individual_pairs = []
        for split_data in all_individual_data.values():
            all_individual_pairs.extend(split_data)

        combined_individual_dataset = {
            "phenomenon": "writeandimprove_individual",
            "description": "Write&Improve 2024 - All individual edit minimal pairs (with continuous edit merging)",
            "version": "individual_edits",
            "pairs": all_individual_pairs
        }

        with open(combined_individual, 'w', encoding='utf-8') as f:
            json.dump(combined_individual_dataset, f, indent=2, ensure_ascii=False)
        print(f"\nSaved combined individual dataset with {len(all_individual_pairs)} pairs")

    print(f"\nConversion complete!")
    print(f"Generated dataset available at: {individual_output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python convert_writeandimprove.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    convert_writeandimprove_dataset(input_dir, output_dir)