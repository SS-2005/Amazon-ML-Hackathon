"""
Model training and evaluation module.
Trains XGBoost model on candidate pairs and optimizes macro-averaged F_0.5 score threshold.
"""

import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time
import pickle
import sys

from features import compute_batch_features, FEATURE_NAMES

def calculate_macro_f05(gt_dict, pred_dict, s1_all_ids):
    """
    Computes macro-averaged F_0.5 score across all S1 entities.
    Singletons (0 true matches, 0 predicted matches) score 1.0.
    """
    f05_scores = []
    
    for s1_id in s1_all_ids:
        true_m = gt_dict.get(s1_id, set())
        pred_m = pred_dict.get(s1_id, set())
        
        if not true_m and not pred_m:
            f05_scores.append(1.0)
            continue
            
        if not pred_m and true_m:
            f05_scores.append(0.0)
            continue
            
        tp = len(true_m.intersection(pred_m))
        fp = len(pred_m - true_m)
        fn = len(true_m - pred_m)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (0.25 * precision + recall) > 0:
            score = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            score = 0.0
            
        f05_scores.append(score)
        
    return float(np.mean(f05_scores))

def train_and_optimize(train_dir, model_save_path, sample_s1_limit=50000):
    """
    Trains XGBoost on training split candidates and finds optimal F_0.5 threshold.
    """
    print(f"Training ML matching model on sample S1 entities (limit={sample_s1_limit:,})...")
    t0 = time.time()
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='8GB';")
    
    s1_p = os.path.join(train_dir, "train_source1.tsv").replace('\\', '/')
    s2_p = os.path.join(train_dir, "train_source2.tsv").replace('\\', '/')
    s3_p = os.path.join(train_dir, "train_source3.tsv").replace('\\', '/')
    gt_p = os.path.join(train_dir, "train_ground_truth.tsv").replace('\\', '/')
    
    # 1. Load sample S1 and Ground Truth
    con.execute(f"""
    CREATE TEMPORARY TABLE s1 AS
    SELECT entity_id, country, business_name, business_address,
           lower(regexp_extract(business_name, '[a-zA-Z0-9]+')) as word1,
           lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) as prefix4,
           regexp_extract(business_address, '[0-9]+') as house_num
    FROM read_csv_auto('{s1_p}', delim='\t', header=True)
    LIMIT {sample_s1_limit};
    """)
    
    con.execute(f"""
    CREATE TEMPORARY TABLE gt AS
    SELECT source1_entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) as s23_id
    FROM read_csv_auto('{gt_p}', delim='\t', header=True)
    WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''
      AND source1_entity_id IN (SELECT entity_id FROM s1);
    """)
    
    # Load S2 and S3 records
    con.execute(f"""
    CREATE TEMPORARY TABLE s23 AS
    SELECT entity_id, country, business_name, business_address,
           lower(regexp_extract(business_name, '[a-zA-Z0-9]+')) as word1,
           lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) as prefix4,
           regexp_extract(business_address, '[0-9]+') as house_num
    FROM (
        SELECT entity_id, country, business_name, business_address FROM read_csv_auto('{s2_p}', delim='\t', header=True)
        UNION ALL
        SELECT entity_id, country, business_name, business_address FROM read_csv_auto('{s3_p}', delim='\t', header=True)
    );
    """)
    
    # Identify high-frequency generic words in S2/S3 (freq > 2000)
    con.execute("""
    CREATE TEMPORARY TABLE frequent_words AS
    SELECT word1, count(*) as cnt
    FROM s23
    WHERE word1 IS NOT NULL AND word1 != ''
    GROUP BY word1
    HAVING count(*) > 2000;
    """)
    
    # Generate blocking candidates
    con.execute("""
    CREATE TEMPORARY TABLE cand AS
    SELECT s1.entity_id as s1_id, s23.entity_id as s23_id,
           s1.business_name as s1_name, s1.business_address as s1_addr,
           s23.business_name as s23_name, s23.business_address as s23_addr
    FROM s1 JOIN s23 ON s1.country = s23.country AND s1.word1 = s23.word1
    WHERE len(s1.word1) >= 3 AND s1.word1 NOT IN (SELECT word1 FROM frequent_words);
    """)
    
    # Mark labels (1 if in GT, 0 otherwise)
    pairs_df = con.execute("""
    SELECT c.*, CASE WHEN g.s23_id IS NOT NULL THEN 1 ELSE 0 END as label
    FROM cand c LEFT JOIN gt g ON c.s1_id = g.s1_id AND c.s23_id = g.s23_id;
    """).df()
    
    print(f"Generated {len(pairs_df):,} training candidate pairs in {time.time()-t0:.2f}s")
    
    # Extract features using fast batch extractor
    print("Extracting features for training pairs...")
    X = compute_batch_features(
        pairs_df['s1_name'].tolist(),
        pairs_df['s1_addr'].tolist(),
        pairs_df['s23_name'].tolist(),
        pairs_df['s23_addr'].tolist()
    )
    y = pairs_df['label'].values
    
    # Train-val split
    s1_unique = pairs_df['s1_id'].unique()
    val_s1 = set(s1_unique[:int(len(s1_unique) * 0.2)])
    
    is_val = pairs_df['s1_id'].isin(val_s1)
    X_train, y_train = X[~is_val], y[~is_val]
    X_val, y_val = X[is_val], y[is_val]
    val_df = pairs_df[is_val].copy()
    
    print(f"Training XGBoost on {len(X_train):,} samples...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4
    )
    model.fit(X_train, y_train)
    
    # Optimize Threshold on Validation Set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_df['prob'] = val_probs
    
    # Load full GT for val S1 set
    gt_val_dict = {}
    gt_rows = con.execute("SELECT s1_id, s23_id FROM gt WHERE s1_id IN (SELECT s1_id FROM cand)").df()
    for _, r in gt_rows.iterrows():
        s1_id = r['s1_id']
        if s1_id not in gt_val_dict:
            gt_val_dict[s1_id] = set()
        gt_val_dict[s1_id].add(r['s23_id'])
        
    best_thresh = 0.65
    best_f05 = -1.0
    
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    print("\n--- Threshold Optimization for Macro F_0.5 ---")
    for t in thresholds:
        pred_dict = {}
        sub_df = val_df[val_df['prob'] >= t]
        for s1_id, s23_id in zip(sub_df['s1_id'], sub_df['s23_id']):
            if s1_id not in pred_dict:
                pred_dict[s1_id] = set()
            pred_dict[s1_id].add(s23_id)
            
        f05 = calculate_macro_f05(gt_val_dict, pred_dict, list(val_s1))
        print(f"Threshold {t:.2f} -> Macro F_0.5: {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
            
    print(f"\nOPTIMAL THRESHOLD = {best_thresh:.2f} with Validation Macro F_0.5 = {best_f05:.4f}")
    
    # Save model and best threshold
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': best_thresh}, f)
        
    print(f"Model saved to {model_save_path}")
    return model, best_thresh

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        train_and_optimize(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python train_eval.py <train_dir> <model_save_path>")
