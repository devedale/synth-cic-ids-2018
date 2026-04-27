#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP2Vec Neural Embeddings module via Native PySpark Skip-gram (Word2Vec).
Features ultra-fast offline Geolocation mapping and strict token namespace prefixing.
"""

from typing import List
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Word2Vec
from pyspark.sql import DataFrame

from pathlib import Path
import pandas as pd

# Advanced Offline GeoIP Lookup leveraging MaxMind GeoLite2
@F.pandas_udf(StringType())
def geoip_region_udf(ips: pd.Series) -> pd.Series:
    import geoip2.database
    import ipaddress
    
    # Path to the downloaded MaxMind DB
    db_path = Path(__file__).resolve().parents[1] / "data" / "intel_cache" / "GeoLite2-Country.mmdb"
    
    # We open the reader once per batch inside the Pandas UDF for massive performance gains
    reader = None
    if db_path.exists():
        try:
            reader = geoip2.database.Reader(str(db_path))
        except Exception:
            pass
            
    def resolve_ip(ip_str):
        if not ip_str or not isinstance(ip_str, str):
            return "UNKNOWN"
            
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private:
                return "PRIVATE_LAN"
            if ip.is_loopback:
                return "LOOPBACK"
            if ip.is_multicast:
                return "MULTICAST"
            if ip.is_reserved:
                return "RESERVED"
        except:
            return "UNKNOWN"
            
        if reader:
            try:
                response = reader.country(ip_str)
                return response.country.iso_code or "UNKNOWN"
            except Exception: # AddressNotFoundError
                return "UNKNOWN"
        else:
            return "UNKNOWN_NO_DB"
            
    res = ips.apply(resolve_ip)
    
    if reader:
        reader.close()
        
    return res


def compute_ip2vec_embeddings(df: DataFrame, context_columns: List[str], vector_size: int = 16) -> DataFrame:
    """
    Transforms network entities into orthogonal text tokens, builds a word array,
    and maps them to a continuous latent space using parallel Skip-gram.
    
    Args:
        df: The native PySpark DataFrame.
        context_columns: Columns representing the categorical context window (e.g. ['Dst Port', 'Protocol']).
        vector_size: Latent dimensions for the generated embedding vector.
    """
    print(f"\n[ip2vec] Applying Prefix Isolation for vectors: {context_columns}")
    
    # Synthesize Region tokens dynamically from IP columns using GeoIP offline lookup
    region_tokens = {"Src Region": "Src IP", "Dst Region": "Dst IP"}
    for region_col, ip_col in region_tokens.items():
        if region_col in context_columns:
            if ip_col not in df.columns:
                print(f"[ip2vec] WARNING: Cannot generate '{region_col}' without '{ip_col}'. Reverting to UNKNOWN.")
                df = df.withColumn(region_col, F.lit("UNKNOWN"))
            else:
                df = df.withColumn(region_col, geoip_region_udf(F.col(ip_col)))
            
    # 2. Strict Prefixing to avoid integer overlaps (e.g., Port 80 != Protocol 80)
    # We dynamically append the column name as a string prefix.
    prefixed_cols = []
    for col_name in context_columns:
        safe_name = col_name.replace(" ", "")
        prefixed_col_name = f"_ip2vec_token_{safe_name}"
        
        # Concat creates completely orthogonal words: e.g. "DstPort_80"
        df = df.withColumn(prefixed_col_name, F.concat(F.lit(f"{safe_name}_"), F.col(col_name).cast("string")))
        prefixed_cols.append(prefixed_col_name)
        
    # 3. Create contextual Sequence Array
    df = df.withColumn("ip2vec_sequence", F.array(*prefixed_cols))
    
    # 4. Native PySpark Skip-gram Neural Word2Vec training
    print("[ip2vec] Training Native PySpark Word2Vec (Skip-gram) distributed model...")
    from configs.settings import RANDOM_SEED

    word2vec = Word2Vec(
        vectorSize=vector_size, 
        minCount=1,
        maxSentenceLength=2,
        inputCol="ip2vec_sequence", 
        outputCol="ip2vec_embeddings",
        windowSize=2,
        numPartitions=4,
        seed=RANDOM_SEED,
    )
    
    model = word2vec.fit(df)
    
    from configs.settings import MODELS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(MODELS_DIR / "ip2vec_model"))
    
    df = model.transform(df)
    
    # 5. Cleanup temporary sequence and prefixed tokens
    cols_to_drop = ["ip2vec_sequence"] + prefixed_cols
    df = df.drop(*cols_to_drop)
    
    print("[ip2vec] Latent Entity Vectors successfully merged into PySpark pipeline.")
    
    return df
