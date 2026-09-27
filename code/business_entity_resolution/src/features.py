"""
Feature extraction module for pairwise string and address similarity.
Uses rapidfuzz C++ backend for high-performance vectorized string matching.
"""

from rapidfuzz import fuzz
import numpy as np
import re

RE_NUM = re.compile(r'\b\d{2,5}\b')

def extract_house_num(addr):
    if not addr:
        return ""
    m = RE_NUM.findall(str(addr))
    return m[0] if m else ""

def compute_pair_features(name1, addr1, name2, addr2):
    """
    Computes a vector of 8 similarity features between two records.
    """
    n1 = str(name1).lower() if name1 else ""
    n2 = str(name2).lower() if name2 else ""
    a1 = str(addr1).lower() if addr1 else ""
    a2 = str(addr2).lower() if addr2 else ""
    
    f_name_ratio = fuzz.ratio(n1, n2) / 100.0
    f_name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    f_name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    
    f_addr_ratio = fuzz.ratio(a1, a2) / 100.0
    f_addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
    
    num1 = extract_house_num(a1)
    num2 = extract_house_num(a2)
    f_house_num_match = 1.0 if (num1 and num2 and num1 == num2) else 0.0
    
    p1 = n1[:4].strip()
    p2 = n2[:4].strip()
    f_prefix_match = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    
    return [
        f_name_ratio,
        f_name_partial,
        f_name_token_sort,
        f_name_token_set,
        f_addr_ratio,
        f_addr_token_set,
        f_house_num_match,
        f_prefix_match
    ]

def compute_batch_features(n1_list, a1_list, n2_list, a2_list):
    """
    Fast batch feature extractor iterating over Python zip lists.
    """
    n = len(n1_list)
    X = np.zeros((n, 8), dtype=np.float32)
    
    for i, (name1, addr1, name2, addr2) in enumerate(zip(n1_list, a1_list, n2_list, a2_list)):
        X[i] = compute_pair_features(name1, addr1, name2, addr2)
        
    return X

FEATURE_NAMES = [
    'name_ratio',
    'name_partial',
    'name_token_sort',
    'name_token_set',
    'addr_ratio',
    'addr_token_set',
    'house_num_match',
    'prefix_match'
]
