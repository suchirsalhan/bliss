# BLISS Dataset Reconstruction Toolkit

This repository contains the open-source toolkit to reconstruct the BLISS dataset introduced in our paper. Due to data redistribution agreements, we cannot provide the processed dataset directly, but this toolkit allows you to reconstruct it from the original sources.

## Overview

The BLISS dataset consists of grammatical minimal pairs derived from learner corpora, enhanced with artificial grammatical errors for contrastive training. The dataset includes:

- **Write&Improve 2024**: Individual edit minimal pairs (63,926 pairs)
- **EFCamDat**: Multi-language learner corpus with artificial error augmentation
- **FCE**: Cambridge First Certificate in English corpus with error analysis

## Requirements

```bash
pip install spacy transformers lemminflect
python -m spacy download en_core_web_sm
```

## Data Sources

You need to obtain the following datasets independently:

### Write&Improve 2024
- **Source**: [Write&Improve Shared Task 2024](https://www.cl.cam.ac.uk/research/nl/bea2024st/)
- **Required files**:
  - `en-writeandimprove2024-final-versions-{train,dev}-sentences.orig`
  - `en-writeandimprove2024-final-versions-{train,dev}-sentences.corr`
  - `en-writeandimprove2024-final-versions-{train,dev}-sentences.m2`
  - `en-writeandimprove2024-final-versions-{train,dev}-sentences.ids`
  - `en-writeandimprove2024-final-versions-m2-essay-info.tsv`

### EFCamDat
- **Source**: [EF Cambridge Open Language Database](https://corpus.mml.cam.ac.uk/efcamdat2/)
- **Note**: Requires academic license and preprocessing with ERRANT

### FCE Dataset
- **Source**: [Cambridge Learner Corpus](https://www.cambridge.org/gb/cambridgeenglish/catalog/cambridge-learner-corpus)
- **Note**: Requires academic license

## Usage

### Step 1: Convert Raw Data to BLIMP Format

```bash
# Convert Write&Improve data
python convert_writeandimprove.py /path/to/writeandimprove/data /path/to/output

# This creates:
# - individual_edits/writeandimprove_{train,dev}_individual_edits.json
```

### Step 2: Generate Artificial Errors

```bash
# Generate for Write&Improve (creates the paper's main dataset)
python generate_artificial_errors.py --datasets writeandimprove --approach precise

# Generate for all datasets
python generate_artificial_errors.py --datasets all --approach precise

# Or use high-coverage approach for better success rates
python generate_artificial_errors.py --datasets all --approach high_coverage
```

### Step 3: Evaluate Models

```bash
# Evaluate any language model on the dataset
python evaluate_model.py gpt2 /path/to/bliss/data --output results/

# Evaluate on specific language speakers
python evaluate_model.py microsoft/DialoGPT-medium /path/to/data --language Spanish

# Evaluate with custom settings
python evaluate_model.py /path/to/local/model /path/to/data --batch-size 8 --tau 0.15
```

### Step 4: Verify Output

The reconstruction should produce files matching these paths from the paper:

```
data/
├── bliss_wi_coverage/
│   ├── writeandimprove_train_precise_individual_edits.json
│   └── writeandimprove_dev_precise_individual_edits.json
├── bliss_efcamdat_coverage/
│   └── [language]_[level]_precise_artificial.json
└── bliss_fce_coverage/
    └── [language]_precise_artificial.json
```

## Dataset Format

Each minimal pair follows this structure:

```json
{
  "sentence_good": "I want to go home.",
  "sentence_bad": "I want go home.",
  "artificial_error": "I want to go to home.",
  "UID": "writeandimprove_individual_train_12345",
  "dataset": "writeandimprove_2024",
  "split": "train",
  "version": "individual",
  "error_type_errant": "M:PREP",
  "sentence_id": "essay_123_sentence_5",
  "essay_id": "essay_123",
  "l1_language": "spanish",
  "auto_cefr": "B1",
  "human_cefr": "B1+",
  "prompt_id": "prompt_456",
  "user_id": "user_789",
  "num_edits": 1,
  "edit_positions": "2:2"
}
```

## Evaluation Metrics

The evaluation script implements the metrics from the paper:

### Headline Metrics (0-100 scale)
- **RP@0**: Random-over-Human Preference (ties count as 0.5)
- **RP@τ**: Random-over-Human Preference with margin threshold τ
- **NGS**: Normalized Gap Score (bounded, symmetric)
- **CPS**: Correct-Preferred Sanity check

### Statistical Testing
- Binomial tests for proportion metrics vs random baseline (50%)
- T-tests for continuous metrics vs baseline
- Significance testing with multiple comparison correction

### Example Output
```
HEADLINE METRICS (0-100 scale):
RP_at_0     :  67.45
RP_at_tau   :  52.13
NGS         :  61.82
CPS         :  87.91

SIGNIFICANCE vs RANDOM (p-values):
RP_at_0     : p = 0.0001 (binomial)
RP_at_tau   : p = 0.0453 (binomial)
NGS         : p = 0.0012 (t-test)
CPS         : p = 0.0003 (binomial)
```

## Key Features

### Error Generation Approaches

1. **Precise Approach**: Generates artificial errors that follow the exact same edit direction and type as the original learner error
   - M:DET (missing determiner) → Delete a different determiner elsewhere
   - U:DET (unnecessary determiner) → Insert a determiner elsewhere
   - R:DET (wrong determiner) → Replace a different determiner

2. **High-Coverage Approach**: Uses more flexible rules and masked language modeling for higher success rates

### Supported Error Types

- **DET**: Determiner errors (the, a, an, this, that, etc.)
- **PREP**: Preposition errors (of, in, on, at, to, for, etc.)
- **NOUN:NUM**: Noun number errors (singular/plural)
- **VERB:TENSE**: Verb tense errors (past/present)
- **VERB:FORM**: Verb form errors (gerund, infinitive, etc.)
- **VERB:SVA**: Subject-verb agreement errors
- **CONJ**: Conjunction errors (and, but, or, because, etc.)

### Quality Filtering

- Sentence length: 3-50 words
- Lexical similarity preservation (Jaccard > 0.7)
- Avoids modifying at the same position as original learner error
- Length change limits (±30%)

## Expected Results

When reconstructed correctly, you should get:

- **Write&Improve Individual**: ~63,926 minimal pairs with ~27% artificial error success rate
- **EFCamDat**: Variable by language/proficiency level
- **FCE**: Variable by first language

## Citation

If you use this toolkit or the reconstructed dataset, please cite our paper:

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install all required packages and spaCy model
2. **File not found**: Verify you have the correct Write&Improve 2024 data files
3. **Low success rates**: The precise approach may have lower success rates (~27%) by design
4. **Memory issues**: Use high-coverage approach or process files individually

### File Structure Expected

```
your_data_directory/
├── en-writeandimprove2024-final-versions-train-sentences.orig
├── en-writeandimprove2024-final-versions-train-sentences.corr
├── en-writeandimprove2024-final-versions-train-sentences.m2
├── en-writeandimprove2024-final-versions-train-sentences.ids
├── en-writeandimprove2024-final-versions-dev-sentences.orig
├── en-writeandimprove2024-final-versions-dev-sentences.corr
├── en-writeandimprove2024-final-versions-dev-sentences.m2
├── en-writeandimprove2024-final-versions-dev-sentences.ids
└── en-writeandimprove2024-final-versions-m2-essay-info.tsv
```

### Success Rate Verification

The artificial error generation success rates should approximately match:

- **Precise approach**: ~27% for Write&Improve
- **High-coverage approach**: ~45-55% for Write&Improve

Lower rates may indicate missing dependencies or data issues.

## License

This toolkit is released under [MIT License]. The original datasets maintain their respective licenses and redistribution terms.

## Contact

For questions about the toolkit or dataset reconstruction, please open an issue or contact [your email].

---

**Note**: This toolkit reconstructs the dataset used in our research. The exact numbers may vary slightly due to preprocessing differences, but should be within ~1-2% of reported figures.