# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Resolver  
**Team Members:** Shivam  
**Submission Date:** September 2026  

---

## 1. Executive Summary

This solution presents an end-to-end, high-performance Machine Learning pipeline for large-scale Business Entity Resolution, resolving 1.73 million Source 1 reference entities against ~10 million candidate records from Source 2 and Source 3 across three countries (US, India, and France). The architecture combines a country-partitioned multi-key inverted index blocking engine with a feature-rich XGBoost classification model. By directly optimizing the decision threshold against the competition's macro-averaged $F_{0.5}$ metric (which penalizes false merges 2× more than false negatives) and implementing strict singleton preservation, our pipeline achieves both exceptional precision and lightning-fast inference (~60,000 pairs/sec) without requiring external lookups.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12.5 million training records and 11.7 million test records revealed several foundational insights:
1. **Intra-Country Strictness:** In the training ground truth, 100.0% of true matches occur strictly within the same country (`US` ↔ `US`, `India` ↔ `India`). Partitioning the entity space by country produces zero cross-country leakage while partitioning the quadratic comparison space by >70%. The test dataset introduces `France`, which must be treated under identical intra-country partition rules.
2. **Singleton Preponderance:** 5.58% of reference entities have zero matches across Source 2 and Source 3. In the macro $F_{0.5}$ evaluation metric, correctly outputting an empty string for singletons yields a perfect 1.0 score, whereas any spurious false-positive match drops the score for that entity directly to 0.0.
3. **Multiplicity & Noise Inconsistencies:** Matched entities average 3.4 matches (up to 11). Business names exhibit widespread abbreviation noise (e.g., `Corp` vs `Corporation`, `Pvt Ltd` vs `Private Limited`), trade names (e.g., `@primemoney` vs `Prime Money Inc`), and transliterations. Addresses exhibit heavy component reordering, municipal numbering format variations, and landmark references.

### 2.2 Solution Strategy
We developed a scalable two-stage hybrid pipeline:
- **Stage 1 (Candidate Generation / Blocking):** A country-partitioned inverted index that extracts multiple deterministic and phonetic-invariant keys per entity, reducing the theoretical 17.3 trillion comparison space to 47.2 million plausible candidate pairs (>99.999% reduction ratio) while preserving high recall.
- **Stage 2 (Feature Engineering & ML Matching):** An 8-dimensional pairwise similarity feature extraction engine leveraging C++ RapidFuzz implementations, followed by an XGBoost binary classifier whose probability threshold is calibrated on held-out validation data to maximize macro $F_{0.5}$.

**Approach Type:** Multi-Key Inverted Index Blocking + Pairwise GBDT Classifier (Hybrid)  
**Core Innovation:** High-frequency word adaptive stopword filtering combined with an element-wise streaming batch inference engine capable of scoring 47+ million candidate pairs in memory in minutes with zero disk thrashing.

---

## 3. Candidate Generation (Blocking)

To avoid quadratic explosion while ensuring true matches are not lost, we employed multi-key blocking partitioned strictly by country:
- **Blocking Keys Used:**
  1. `Key 1 (Country + Primary Token):` Country label combined with the first significant alphanumeric token of length $\ge 3$ characters, excluding a dynamically identified set of high-frequency generic stopwords (e.g., `the`, `group`, `enterprises`, `services`, `center`).
  2. `Key 2 (Country + House Number + Prefix):` Country label combined with normalized street address numerical digits (length $\ge 2$) and the first 3 characters of the business name prefix.
- **Candidate Pairs Generated:** 47,238,366 candidate pairs across 1,732,544 test Source 1 entities (capped at 50 candidates per entity).
- **Preserving True Matches:** Combining token matching with numerical street address matching guarantees that entities with altered prefixes (e.g., trade names or legal additions like `The`) are still captured via their address house numbers.

---

## 4. Matching Model

### Features Used:
1. `name_ratio:` Normalized Levenshtein distance ratio between full business names.
2. `name_partial_ratio:` Substring matching ratio, capturing cases where one source contains an abbreviation or trade handle.
3. `name_token_sort_ratio:` Token sort ratio, invariant to word transposition (e.g., `Barbershop Orelee` vs `Orelee Barbershop`).
4. `name_token_set_ratio:` Token set ratio, invariant to legal suffixes (e.g., `Prime Money Inc.` vs `Prime Money`).
5. `addr_ratio:` Full address character similarity ratio.
6. `addr_token_set_ratio:` Address token set overlap, robust against component reordering (e.g., city/state appearing before street name).
7. `house_num_match:` Binary indicator flag ($1.0$ if numeric street numbers match exactly, $0.0$ otherwise).
8. `prefix_match:` Binary indicator flag for matching 4-character name prefixes.

### Model Architecture & Training:
- **Model Type:** XGBoost Classifier (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`).
- **Training Data:** Candidate pairs generated from a sample split of `train_source1.tsv` with binary labels derived from `train_ground_truth.tsv`.
- **Threshold Selection:** Grid search over classification thresholds $\tau \in [0.45, 0.85]$ on a 20% held-out validation split directly evaluating the competition macro $F_{0.5}$ metric:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Optimal threshold established: **$\tau^* = 0.65$**, heavily prioritizing precision to prevent destructive false merges.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** ~0.84 on validation split.
- **Common False Positives:** Multiple branches or distinct franchises of a commercial chain located in the same city/state (e.g., different branches of the same bank or retail chain) where address differences are subtle.
- **Common False Negatives:** Severe address omissions where one source records only a landmark while the reference source records only municipal district numbers.

---

## 6. Conclusion

Our solution delivers an industrial-grade, reproducible Business Entity Resolution system capable of processing tens of millions of records within minutes. By enforcing country isolation, multi-key blocking, and precision-targeted threshold tuning, the pipeline maximizes $F_{0.5}$ while maintaining full compliance with all data integrity and submission constraints.

---

## Appendix

### A. Code Artefacts
The complete runnable code is organized inside `code/business_entity_resolution/`:
- `src/blocking.py`: Multi-key country-partitioned candidate generator.
- `src/features.py`: C++ backed RapidFuzz feature extraction module.
- `src/train_eval.py`: Model training and $F_{0.5}$ threshold calibration.
- `src/predict.py`: Vectorized streaming batch inference generating `output/matching_results.tsv`.
- `src/main.py`: End-to-end reproduction entry point.
- `requirements.txt`: Pinned environment specifications.
- `README.md`: Step-by-step reproduction instructions.

### B. Execution Verification
All submission files are verified using the official validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Result: **`PASS — no blocking issues found. Safe to submit.`**
