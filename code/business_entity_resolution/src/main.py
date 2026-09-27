"""
Main pipeline execution entry point for Business Entity Resolution.
Runs end-to-end: Blocking -> Model Training -> Threshold Optimization -> Inference -> Output Validation.
"""

import os
import sys
import time

from blocking import run_blocking
from train_eval import train_and_optimize
from predict import run_prediction

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    dataset_dir = os.path.join(base_dir, "dataset")
    output_dir = os.path.join(base_dir, "output")
    models_dir = os.path.join(base_dir, "models")
    
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    
    test_s1 = os.path.join(test_dir, "test_source1.tsv")
    test_s2 = os.path.join(test_dir, "test_source2.tsv")
    test_s3 = os.path.join(test_dir, "test_source3.tsv")
    
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    model_path = os.path.join(models_dir, "xgboost_model.pkl")
    
    print("==========================================================================")
    print("      AMAZON ML CHALLENGE 2026 - BUSINESS ENTITY RESOLUTION PIPELINE     ")
    print("==========================================================================")
    t_start = time.time()
    
    # Step 1: Run candidate generation (blocking) on test set
    print("\n--- STEP 1: CANDIDATE GENERATION (BLOCKING) ---")
    run_blocking(test_s1, test_s2, test_s3, cand_path, max_cands_per_s1=50)
    
    # Step 2: Train matching model and optimize F_0.5 threshold on training set
    print("\n--- STEP 2: MODEL TRAINING & THRESHOLD OPTIMIZATION ---")
    if not os.path.exists(model_path):
        train_and_optimize(train_dir, model_path, sample_s1_limit=100000)
    else:
        print(f"Using pre-trained model found at {model_path}")
        
    # Step 3: Run inference on test set candidates to produce matching_results.tsv
    print("\n--- STEP 3: MATCHING MODEL INFERENCE ---")
    run_prediction(test_s1, test_s2, test_s3, cand_path, model_path, matching_path)
    
    print("\n==========================================================================")
    print(f"Pipeline executed successfully in {time.time()-t_start:.2f} seconds!")
    print(f"Leaderboard file: {matching_path}")
    print(f"Candidate file:   {cand_path}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
