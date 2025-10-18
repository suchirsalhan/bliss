import json
from pathlib import Path
import argparse

def check_file_structure(base_path):
    base = Path(base_path)
    expected_files = {
        'individual_edits': [
            'writeandimprove_train_individual_edits.json',
            'writeandimprove_dev_individual_edits.json',
            'writeandimprove_all_individual_edits.json'
        ],
        'sentence_level': [
            'writeandimprove_train_sentence_level.json',
            'writeandimprove_dev_sentence_level.json',
            'writeandimprove_all_sentence_level.json'
        ]
    }

    print("Checking file structure...")
    for folder, files in expected_files.items():
        folder_path = base / folder
        if not folder_path.exists():
            print(f"❌ Missing folder: {folder_path}")
            continue

        print(f"✅ Found folder: {folder}")
        for file in files:
            file_path = folder_path / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ Missing: {file}")

def verify_json_format(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'pairs' in data:
            pairs = data['pairs']
        elif isinstance(data, list):
            pairs = data
        else:
            return False, "Invalid JSON structure"

        if not pairs:
            return False, "No pairs found"

        # check first pair format
        first_pair = pairs[0]
        required_fields = ['sentence_good', 'sentence_bad', 'error_type_errant']
        for field in required_fields:
            if field not in first_pair:
                return False, f"Missing field: {field}"

        return True, f"{len(pairs)} pairs"

    except Exception as e:
        return False, str(e)

def check_artificial_errors(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and 'pairs' in data:
            pairs = data['pairs']
        elif isinstance(data, list):
            pairs = data
        else:
            return 0, 0

        total_pairs = len(pairs)
        artificial_count = sum(1 for pair in pairs if 'artificial_error' in pair and pair['artificial_error'])

        return total_pairs, artificial_count

    except Exception as e:
        return 0, 0

def verify_datasets(base_path):
    base = Path(base_path)

    print("\nVerifying dataset format and statistics...")

    # check individual edits
    individual_train = base / 'individual_edits' / 'writeandimprove_train_individual_edits.json'
    if individual_train.exists():
        valid, info = verify_json_format(individual_train)
        if valid:
            print(f"✅ Train individual edits: {info}")
        else:
            print(f"❌ Train individual edits: {info}")

    # check if artificial errors were generated
    artificial_path = Path(base_path).parent / 'bliss_wi_coverage'
    if artificial_path.exists():
        print(f"\nChecking artificial error generation...")

        train_artificial = artificial_path / 'writeandimprove_train_precise_individual_edits.json'
        if train_artificial.exists():
            total, artificial = check_artificial_errors(train_artificial)
            success_rate = (artificial / total * 100) if total > 0 else 0
            print(f"✅ Train artificial errors: {artificial}/{total} ({success_rate:.1f}%)")

            # verify expected success rate
            if 25 <= success_rate <= 30:
                print("✅ Success rate matches expected range (25-30%)")
            else:
                print(f"⚠️  Success rate outside expected range (25-30%)")
        else:
            print("❌ No artificial errors found. Run generate_artificial_errors.py")
    else:
        print("❌ No artificial error directory found")

def main():
    parser = argparse.ArgumentParser(description="Verify BLISS dataset reconstruction")
    parser.add_argument("data_path", help="Path to reconstructed dataset")
    args = parser.parse_args()

    print("BLISS Dataset Verification")
    print("=" * 30)

    check_file_structure(args.data_path)
    verify_datasets(args.data_path)

    print(f"\nExpected final statistics:")
    print(f"- Individual pairs: ~63,926")
    print(f"- Sentence-level pairs: ~25,307")
    print(f"- Artificial error success rate: ~27%")

if __name__ == "__main__":
    main()