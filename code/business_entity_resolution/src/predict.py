"""
Inference module for Business Entity Resolution.
Ultra-fast streaming batch inference:
Loads S1 and S2/S3 in memory dictionaries,
streams candidate_pairs.tsv in batches,
computes RapidFuzz features via C++ backend,
predicts matches using trained XGBoost model,
and outputs matching_results.tsv adhering strictly to all submission constraints.
"""

import os
import sys
import time
import pickle
import numpy as np

# Ensure local imports work
sys.path.append(os.path.dirname(__file__))
from features import compute_batch_features

def run_prediction(s1_tsv_path, s2_tsv_path, s3_tsv_path,
                   candidate_tsv_path, model_path, output_matching_path,
                   batch_pairs_size=150_000):
    print("==========================================================================")
    print("                STARTING MATCHING MODEL INFERENCE PIPELINE               ")
    print("==========================================================================")
    t_start = time.time()
    
    # 1. Load trained model and threshold
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        threshold = data['threshold']
    print(f"Model loaded successfully | Decision threshold: {threshold:.2f}")
    
    # 2. Load S1 text details into dict
    print(f"Loading S1 reference records from {s1_tsv_path}...")
    t0 = time.time()
    s1_dict = {}
    s1_ordered_ids = []
    with open(s1_tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 3:
                eid = parts[0]
                s1_dict[eid] = (parts[1], parts[2])
                s1_ordered_ids.append(eid)
            elif len(parts) == 2:
                eid = parts[0]
                s1_dict[eid] = (parts[1], '')
                s1_ordered_ids.append(eid)
    print(f"Loaded {len(s1_dict):,} S1 entities in {time.time()-t0:.2f}s")
    
    # 3. Load S2 and S3 text details into dict
    print("Loading S2 and S3 candidate records...")
    t1 = time.time()
    s23_dict = {}
    for p in [s2_tsv_path, s3_tsv_path]:
        print(f"  Reading {p}...")
        with open(p, 'r', encoding='utf-8') as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 3:
                    s23_dict[parts[0]] = (parts[1], parts[2])
                elif len(parts) == 2:
                    s23_dict[parts[0]] = (parts[1], '')
    print(f"Loaded {len(s23_dict):,} S2/S3 candidate records in {time.time()-t1:.2f}s")
    
    # 4. Stream through candidate_pairs.tsv and score in batches
    print(f"\nStreaming candidates from {candidate_tsv_path} (batch size: {batch_pairs_size:,} pairs)...")
    matches_dict = {} # s1_id -> list of matched s23_ids
    
    buffer_s1_ids = []
    buffer_cand_ids = []
    n1_buf = []
    a1_buf = []
    n2_buf = []
    a2_buf = []
    
    total_pairs_scored = 0
    total_matches_found = 0
    batch_idx = 0
    t_stream_start = time.time()
    
    def process_buffer():
        nonlocal total_pairs_scored, total_matches_found, batch_idx
        if not buffer_s1_ids:
            return
        batch_idx += 1
        n_pairs = len(buffer_s1_ids)
        total_pairs_scored += n_pairs
        
        # Compute features
        t_b0 = time.time()
        X = compute_batch_features(n1_buf, a1_buf, n2_buf, a2_buf)
        
        # Predict
        probs = model.predict_proba(X)[:, 1]
        mask = probs >= threshold
        n_matches = int(mask.sum())
        total_matches_found += n_matches
        
        # Collect matches
        for i in np.where(mask)[0]:
            sid = buffer_s1_ids[i]
            cid = buffer_cand_ids[i]
            if sid not in matches_dict:
                matches_dict[sid] = []
            matches_dict[sid].append(cid)
            
        elapsed = time.time() - t_b0
        rate = n_pairs / elapsed if elapsed > 0 else 0
        print(f"  Batch {batch_idx:3d}: {n_pairs:7,d} pairs | {n_matches:5,d} matches | {rate:7,.0f} pairs/sec | Total scored: {total_pairs_scored:10,d} | Matches: {total_matches_found:8,d}", flush=True)
        
        # Clear buffer
        buffer_s1_ids.clear()
        buffer_cand_ids.clear()
        n1_buf.clear()
        a1_buf.clear()
        n2_buf.clear()
        a2_buf.clear()

    with open(candidate_tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            s1_id, _, cand_str = line.rstrip('\n').partition('\t')
            if not cand_str or s1_id not in s1_dict:
                continue
            cand_ids = cand_str.split(',')
            s1_name, s1_addr = s1_dict[s1_id]
            
            for cid in cand_ids:
                if cid in s23_dict:
                    s23_name, s23_addr = s23_dict[cid]
                    buffer_s1_ids.append(s1_id)
                    buffer_cand_ids.append(cid)
                    n1_buf.append(s1_name)
                    a1_buf.append(s1_addr)
                    n2_buf.append(s23_name)
                    a2_buf.append(s23_addr)
                    
                    if len(buffer_s1_ids) >= batch_pairs_size:
                        process_buffer()
                        
    # Process remaining pairs
    process_buffer()
    
    print(f"\nScoring completed in {time.time()-t_stream_start:.2f}s!")
    print(f"Total candidate pairs scored: {total_pairs_scored:,}")
    print(f"Total matched pairs found:    {total_matches_found:,}")
    
    # 5. Write matching_results.tsv with strict compliance to submission format
    print(f"\nWriting final matching results to {output_matching_path}...")
    os.makedirs(os.path.dirname(output_matching_path), exist_ok=True)
    
    singletons = 0
    multi_matched = 0
    with open(output_matching_path, 'w', encoding='utf-8', newline='\n') as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_ordered_ids:
            if eid in matches_dict and matches_dict[eid]:
                # deduplicate preserving order
                unique_matches = list(dict.fromkeys(matches_dict[eid]))
                f_out.write(f"{eid}\t{','.join(unique_matches)}\n")
                multi_matched += 1
            else:
                f_out.write(f"{eid}\t\n")
                singletons += 1
                
    total_written = singletons + multi_matched
    print("==========================================================================")
    print(f"MATCHING GENERATION COMPLETE!")
    print(f"  Total S1 rows written: {total_written:,} (Expected: {len(s1_ordered_ids):,})")
    print(f"  Singletons (no match): {singletons:,} ({singletons/total_written*100:.2f}%)")
    print(f"  Entities with matches: {multi_matched:,} ({multi_matched/total_written*100:.2f}%)")
    print(f"  Total pipeline time:   {time.time()-t_start:.2f}s")
    print(f"  Output path:           {output_matching_path}")
    print("==========================================================================")
    
    return output_matching_path

if __name__ == "__main__":
    if len(sys.argv) >= 7:
        run_prediction(sys.argv[1], sys.argv[2], sys.argv[3],
                       sys.argv[4], sys.argv[5], sys.argv[6])
    else:
        print("Usage: python predict.py <s1.tsv> <s2.tsv> <s3.tsv> <candidate_pairs.tsv> <model.pkl> <matching_results.tsv>")
