import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from configs.settings import TARGET_BENIGN_RATIO, RANDOM_SEED


def get_dataset(df: DataFrame, strategy: str = "raw", target_benign_ratio: float = TARGET_BENIGN_RATIO) -> DataFrame:
    """
    SOTA Data Loader: Lazily applies the selected Matrix shape strategy 
    without duplicating data on disk.
    
    Args:
        df: Base DataFrame natively loaded from parquets
        strategy: "raw", "unsupervised", "binary_collapse", "undersample_majority"
        target_benign_ratio: Explicit ratio configuration for undersampling mode
    """
    from configs.settings import USE_PCA, USE_IP2VEC, NET_ENTITIES
    
    # Select Dimensional Vector Logic (PCA vs Standard)
    if "pca_features" in df.columns:
        if USE_PCA:
            df = df.drop("features").withColumnRenamed("pca_features", "features")
        else:
            df = df.drop("pca_features")
            
    # Conditional Embeddings Exclusion
    if not USE_IP2VEC:
        # Drop categorical networking paths if legacy non-embedding architecture is active
        found_entities = [c for c in NET_ENTITIES if c in df.columns]
        if found_entities:
            df = df.drop(*found_entities)
    
    if strategy == "raw":
        return df
        
    elif strategy == "unsupervised":
        # Purely anomalous-free (or exclusively anomalous) dataset definition for Autoencoders
        return df.filter(F.col("Label") == "Benign")
        
    elif strategy == "binary_collapse":
        # Multi-class to 2-Class reduction
        return df.withColumn(
            "Label", 
            F.when(F.col("Label") != "Benign", "Attack").otherwise("Benign")
        )
        
    elif strategy == "undersample_majority":
        # Stratified Native Downsampling mechanism using PySpark
        # Multi-class detection: We keep the original labels but compute the sampling fractions 
        # based on 'benign' vs 'all_attacks'.
        
        # 1. Identify which rows are benign vs attacks (case-insensitive + trim)
        df_binary = df.withColumn(
            "is_benign", 
            F.when(F.trim(F.lower(F.col("Label"))) == "benign", True).otherwise(False)
        )
        
        # 2. Get counts
        counts = df_binary.groupBy("is_benign").count().collect()
        count_dict = {row['is_benign']: row['count'] for row in counts}
        
        b_count = count_dict.get(True, 0)
        a_count = count_dict.get(False, 0) # Total count of all attacks combined
        
        if b_count == 0 or a_count == 0:
            return df # Cannot balance if one side is missing
            
        current_ratio = b_count / (b_count + a_count)
        
        # 3. Compute sampling fraction for Benign rows
        if current_ratio > target_benign_ratio:
            desired_b_count = a_count * (target_benign_ratio / (1.0 - target_benign_ratio))
            b_sample_frac = min(1.0, desired_b_count / b_count)
        else:
            b_sample_frac = 1.0 # Already weighted towards attacks or balanced
            
        # 4. Filter and sample deterministically using row hashing
        # PySpark's sampleBy relies on partition order which is non-deterministic.
        # We generate a deterministic pseudo-random float [0.0, 1.0] from row contents.
        df_binary = df_binary.withColumn(
            "_rand_val", 
            F.abs(F.xxhash64(F.lit(RANDOM_SEED), *df.columns)) / F.lit(9223372036854775807)
        )
        
        sampled_df = df_binary.filter(
            (F.col("is_benign") == False) | 
            ((F.col("is_benign") == True) & (F.col("_rand_val") <= b_sample_frac))
        )
        
        return sampled_df.drop("is_benign", "_rand_val")
        
    else:
        raise ValueError(f"Unknown Strategy: {strategy}. Choose from 'raw', 'unsupervised', 'binary_collapse', 'undersample_majority'.")
