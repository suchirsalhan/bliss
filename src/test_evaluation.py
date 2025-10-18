#!/usr/bin/env python3
"""
Test script for the evaluation module.
"""

import sys
from pathlib import Path
import json

def test_evaluation_imports():
    """Test that evaluation modules can be imported."""
    try:
        sys.path.append(str(Path(__file__).parent))

        import evaluate_model
        print("✓ evaluate_model import successful")

        # Test that key functions exist
        assert hasattr(evaluate_model, 'evaluate_model_on_dataset')
        assert hasattr(evaluate_model, 'ModelEvaluator')
        assert hasattr(evaluate_model, 'compute_metrics')
        print("✓ All required functions present")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing."""
    sample_data = {
        "pairs": [
            {
                "sentence_good": "I want to go home.",
                "sentence_bad": "I want go home.",
                "artificial_error": "I want to go to home.",
                "l1_language": "Spanish",
                "error_type_errant": "M:PREP",
                "UID": "test_1"
            },
            {
                "sentence_good": "She is a good student.",
                "sentence_bad": "She is good student.",
                "artificial_error": "She is an good student.",
                "l1_language": "Chinese",
                "error_type_errant": "M:DET",
                "UID": "test_2"
            }
        ]
    }

    test_file = "/tmp/claude/test_bliss_data.json"
    Path("/tmp/claude").mkdir(exist_ok=True)

    with open(test_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

    return test_file

def test_metrics_computation():
    """Test metrics computation with sample data."""
    try:
        from evaluate_model import ItemTriplet, compute_metrics

        # Create sample items
        items = [
            ItemTriplet(
                file="test.json", idx=0, id="test_1", l1="Spanish",
                good="I want to go home.", bad="I want go home.", art="I want to go to home.",
                bpt_good=3.2, bpt_bad=4.1, bpt_art=4.5
            ),
            ItemTriplet(
                file="test.json", idx=1, id="test_2", l1="Chinese",
                good="She is a good student.", bad="She is good student.", art="She is an good student.",
                bpt_good=2.8, bpt_bad=3.6, bpt_art=3.9
            )
        ]

        # Compute metrics
        results = compute_metrics(items, tau=0.1)

        # Check that all expected metrics are present
        assert "headline_metrics" in results
        assert "diagnostic_metrics" in results

        headline = results["headline_metrics"]
        required_metrics = ["RP_at_0", "RP_at_tau", "NGS", "CPS"]
        for metric in required_metrics:
            assert metric in headline
            assert 0 <= headline[metric] <= 100  # All metrics should be 0-100 scale

        print("✓ Metrics computation test passed")
        print(f"  Sample RP@0: {headline['RP_at_0']:.2f}")
        print(f"  Sample NGS:  {headline['NGS']:.2f}")

        return True

    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the full evaluation pipeline (without actual model loading)."""
    print("Testing evaluation pipeline (mock)...")

    # This test would require actual model loading which is expensive
    # So we just test the data loading and preprocessing parts

    try:
        from evaluate_model import load_dataset_files, iter_json_records

        # Create test data
        test_file = create_sample_data()

        # Test file discovery
        files = load_dataset_files(str(Path(test_file).parent))
        assert len(files) > 0
        print(f"✓ Found {len(files)} test files")

        # Test JSON record iteration
        records = list(iter_json_records(test_file))
        assert len(records) == 2
        print(f"✓ Loaded {len(records)} test records")

        # Verify record structure
        for record in records:
            assert "sentence_good" in record
            assert "sentence_bad" in record
            assert "artificial_error" in record
            assert "l1_language" in record

        print("✓ Pipeline test passed")
        return True

    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False

def main():
    print("BLISS Evaluation Module Test")
    print("=" * 40)

    tests = [
        test_evaluation_imports,
        test_metrics_computation,
        test_full_pipeline,
    ]

    passed = 0
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1

    print(f"\n" + "=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("🎉 All tests passed! Evaluation module is ready to use.")
        print("\nNext steps:")
        print("1. Ensure you have the reconstructed BLISS dataset")
        print("2. Run: python evaluate_model.py gpt2 /path/to/data --output results/")
    else:
        print("❌ Some tests failed. Check error messages above.")

if __name__ == "__main__":
    main()