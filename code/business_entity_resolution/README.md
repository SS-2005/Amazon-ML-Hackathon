# Business Entity Resolution Solution

This directory contains the self-contained, reproducible pipeline for the Business Entity Resolution Challenge (Amazon ML Challenge 2026).

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── blocking.py      # Country-partitioned multi-key candidate generator
│   ├── features.py      # RapidFuzz pairwise string & address similarity extractor
│   ├── train_eval.py    # XGBoost model training and macro F_0.5 threshold optimizer
│   ├── predict.py      # Test inference generating output/matching_results.tsv
│   └── main.py         # End-to-end pipeline entry point
├── README.md            # Execution instructions (this file)
└── requirements.txt     # Environment dependencies
```

## How to Reproduce End-to-End

### 1. Prerequisites
Python 3.8+ with installed dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution
Run the full end-to-end pipeline from the `student_resource/` directory:
```bash
python code/business_entity_resolution/src/main.py
```

This will execute:
1. Candidate blocking on test data -> `output/candidate_pairs.tsv`
2. XGBoost training & $F_{0.5}$ threshold optimization -> `models/xgboost_model.pkl`
3. Inference on test candidates -> `output/matching_results.tsv`

### 3. Validation
Validate output formatting compliance before portal submission:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
