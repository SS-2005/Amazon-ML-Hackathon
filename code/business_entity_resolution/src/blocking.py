"""
Candidate generation (blocking) module for Business Entity Resolution.
Partitioned strictly by country, using multi-key inverted index blocking with high-frequency word filtering.
Produces candidate_pairs.tsv complying with all challenge requirements.
"""

import duckdb
import os
import time
import sys

def run_blocking(s1_tsv_path, s2_tsv_path, s3_tsv_path, output_candidate_path, max_cands_per_s1=50):
    """
    Runs multi-key country-partitioned candidate generation in DuckDB.
    Writes candidate_pairs.tsv to output_candidate_path.
    """
    print(f"Starting blocking engine for S1: {s1_tsv_path}")
    t0 = time.time()
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='8GB';")
    
    s1_p = s1_tsv_path.replace('\\', '/')
    s2_p = s2_tsv_path.replace('\\', '/')
    s3_p = s3_tsv_path.replace('\\', '/')
    out_p = output_candidate_path.replace('\\', '/')
    
    os.makedirs(os.path.dirname(output_candidate_path), exist_ok=True)
    
    # 1. Load S1
    print("Reading S1 records...")
    con.execute(f"""
    CREATE TEMPORARY TABLE s1 AS
    SELECT entity_id, country, business_name, business_address,
           lower(regexp_extract(business_name, '[a-zA-Z0-9]+')) as word1,
           lower(substr(regexp_replace(business_name, '[^a-zA-Z0-9]', '', 'g'), 1, 4)) as prefix4,
           regexp_extract(business_address, '[0-9]+') as house_num
    FROM read_csv_auto('{s1_p}', delim='\t', header=True);
    """)
    s1_cnt = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    print(f"Loaded {s1_cnt:,} S1 records.")
    
    # 2. Load S2 and S3
    print("Reading S2 and S3 records...")
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
    s23_cnt = con.execute("SELECT count(*) FROM s23").fetchone()[0]
    print(f"Loaded {s23_cnt:,} S2/S3 records.")
    
    # 3. Identify high-frequency generic words in S2/S3 (freq > 2000)
    print("Filtering out high-frequency generic stopwords...")
    con.execute("""
    CREATE TEMPORARY TABLE frequent_words AS
    SELECT word1, count(*) as cnt
    FROM s23
    WHERE word1 IS NOT NULL AND word1 != ''
    GROUP BY word1
    HAVING count(*) > 2000;
    """)
    
    # 4. Block candidates using Country + Word1 (len >= 3, excluding frequent_words)
    print("Generating candidate matches (Filtered Word1 match)...")
    con.execute("""
    CREATE TEMPORARY TABLE cand_w1 AS
    SELECT s1.entity_id as s1_id, s23.entity_id as s23_id
    FROM s1 JOIN s23 ON s1.country = s23.country AND s1.word1 = s23.word1
    WHERE len(s1.word1) >= 3
      AND s1.word1 NOT IN (SELECT word1 FROM frequent_words);
    """)
    
    # 5. Block candidates using Country + House Num + 3-char Name Prefix
    print("Generating candidate matches (House Num + Name Prefix)...")
    con.execute("""
    CREATE TEMPORARY TABLE cand_hn AS
    SELECT s1.entity_id as s1_id, s23.entity_id as s23_id
    FROM s1 JOIN s23 ON s1.country = s23.country AND s1.house_num = s23.house_num AND s1.prefix4[1:3] = s23.prefix4[1:3]
    WHERE len(s1.house_num) >= 2 AND len(s1.prefix4) >= 3;
    """)
    
    # 6. Union candidate sets & rank to limit max candidates per S1
    print("Merging and ranking candidates...")
    con.execute("""
    CREATE TEMPORARY TABLE cand_all AS
    SELECT s1_id, s23_id FROM cand_w1
    UNION ALL
    SELECT s1_id, s23_id FROM cand_hn;
    """)
    
    con.execute(f"""
    CREATE TEMPORARY TABLE cand_ranked AS
    SELECT s1_id, s23_id,
           row_number() OVER (PARTITION BY s1_id ORDER BY s23_id) as rn
    FROM cand_all;
    """)
    
    # 7. Aggregate candidates into comma-separated list per S1 entity
    print("Aggregating candidate pairs per S1 entity...")
    con.execute(f"""
    CREATE TEMPORARY TABLE cand_agg AS
    SELECT s1_id as source1_entity_id,
           string_agg(s23_id, ',') as candidate_entity_ids
    FROM cand_ranked
    WHERE rn <= {max_cands_per_s1}
    GROUP BY s1_id;
    """)
    
    # 8. Left join with all original S1 entities to guarantee 100% coverage (singletons get empty string)
    print("Writing candidate_pairs.tsv...")
    con.execute(f"""
    COPY (
        SELECT s1.entity_id as source1_entity_id,
               coalesce(ca.candidate_entity_ids, '') as candidate_entity_ids
        FROM s1 LEFT JOIN cand_agg ca ON s1.entity_id = ca.source1_entity_id
        ORDER BY s1.entity_id
    ) TO '{out_p}' (HEADER, DELIMITER '\t');
    """)
    
    t_end = time.time()
    print(f"Candidate generation complete! Output written to {output_candidate_path} in {t_end-t0:.2f}s")
    
    return output_candidate_path

if __name__ == "__main__":
    if len(sys.argv) >= 5:
        run_blocking(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Usage: python blocking.py <s1.tsv> <s2.tsv> <s3.tsv> <out_candidate.tsv>")
