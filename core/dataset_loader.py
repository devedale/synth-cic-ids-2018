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
        df_binary = df.withColumn(
            "Label", 
            F.when(F.col("Label") != "Benign", "Attack").otherwise("Benign")
        )
        
        # In reality, PySpark sampleBy works perfectly with fractions dict based on string categories!
        # First we need frequencies (costs an action, but unavoidable for exact ratio mapping)
        fractions = {}
        total_counts = df_binary.groupBy("Label").count().collect()
        count_dict = {row['Label']: row['count'] for row in total_counts}
        
        b_count = count_dict.get('Benign', 0)
        a_count = count_dict.get('Attack', 0)
        
        if b_count == 0 or a_count == 0:
            return df_binary # Cannot balance if one class is entirely missing
            
        current_ratio = b_count / (b_count + a_count)
        
        if current_ratio > target_benign_ratio:
            # Benign is majority
            desired_b_count = a_count * (target_benign_ratio / (1.0 - target_benign_ratio))
            fractions["Benign"] = min(1.0, desired_b_count / b_count)
            fractions["Attack"] = 1.0
        else:
            # Attack is majority
            desired_a_count = b_count * ((1.0 - target_benign_ratio) / target_benign_ratio)
            fractions["Attack"] = min(1.0, desired_a_count / a_count)
            fractions["Benign"] = 1.0
            
        return df_binary.sampleBy("Label", fractions, seed=RANDOM_SEED)
        
    else:
        raise ValueError(f"Unknown Strategy: {strategy}. Choose from 'raw', 'unsupervised', 'binary_collapse', 'undersample_majority'.")
