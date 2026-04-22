import ipaddress
import random
from pyspark.sql import DataFrame, SparkSession, Row
import pyspark.sql.functions as F

def build_translation_table(df: DataFrame, malicious_ips_pool: list, benign_ips_pool: list, label_col: str = "Label") -> dict:
    """
    Step 1 & 2: Pre-computation & The Translation Table Engine.
    Identifies all unique IPs and maps them to synthetic IPs 
    ensuring 1:1 mapping and Private/Public & Malicious/Benign segregation.
    """
    # 1. Identify all unique IPs
    all_ips_df = df.select("Src IP").union(df.select("Dst IP")).distinct()
    unique_ips = [row[0] for row in all_ips_df.collect() if row[0]]
    
    # 2. Identify attackers
    malicious_srcs = set()
    if label_col in df.columns:
        attack_cond = (F.trim(F.lower(F.col(label_col))) != "benign")
        malicious_srcs = {row[0] for row in df.filter(attack_cond).select("Src IP").distinct().collect() if row[0]}
        
    ip_map = {}
    
    # Pools for popping (Category 1 & 2)
    avail_malicious = set(malicious_ips_pool) if malicious_ips_pool else {"198.51.100.1"}
    avail_benign = set(benign_ips_pool) if benign_ips_pool else {"8.8.8.8"}
    
    priv_malicious_counter = 1
    priv_benign_counter = 1
    
    def is_private(ip_str):
        try:
            return ipaddress.ip_address(ip_str).is_private
        except:
            return False
            
    for ip in unique_ips:
        if is_private(ip):
            if ip in malicious_srcs:
                # Category 3: Private Malicious -> 192.168.x.x/16
                synth_ip = f"192.168.{(priv_malicious_counter >> 8) & 255}.{priv_malicious_counter & 255}"
                priv_malicious_counter += 1
                ip_map[ip] = synth_ip
            else:
                # Category 4: Private Benign -> 10.x.x.x/8
                synth_ip = f"10.{(priv_benign_counter >> 16) & 255}.{(priv_benign_counter >> 8) & 255}.{priv_benign_counter & 255}"
                priv_benign_counter += 1
                ip_map[ip] = synth_ip
        else:
            if ip in malicious_srcs:
                # Category 1: Public Malicious
                try:
                    ip_map[ip] = avail_malicious.pop()
                except KeyError:
                    ip_map[ip] = f"198.51.100.{random.randint(1,254)}" # Fallback
            else:
                # Category 2: Public Benign
                try:
                    ip_map[ip] = avail_benign.pop()
                except KeyError:
                    ip_map[ip] = f"203.0.113.{random.randint(1,254)}" # Fallback

    if "" not in ip_map:
        ip_map[""] = ""
        
    return ip_map

def apply_ip_translation(spark: SparkSession, df: DataFrame, ip_map: dict) -> DataFrame:
    """
    Step 3: Fast Broadcast Mapping.
    Instead of F.create_map which hits Java 64KB method limits, we broadcast 
    a small DataFrame and perform a join, achieving the same O(1)-like 
    distributed mapping without row-by-row python UDFs.
    """
    if not ip_map:
        return df
        
    mapping_rows = [Row(original_ip=k, synthetic_ip=v) for k, v in ip_map.items()]
    mapping_df = spark.createDataFrame(mapping_rows)
    
    # Map Src IP
    df_processed = df.join(
        F.broadcast(mapping_df).withColumnRenamed("original_ip", "src_orig").withColumnRenamed("synthetic_ip", "src_synth"),
        df["Src IP"] == F.col("src_orig"),
        "left"
    ).withColumn("Src IP", F.coalesce(F.col("src_synth"), F.col("Src IP"))).drop("src_orig", "src_synth")
    
    # Map Dst IP
    df_processed = df_processed.join(
        F.broadcast(mapping_df).withColumnRenamed("original_ip", "dst_orig").withColumnRenamed("synthetic_ip", "dst_synth"),
        df_processed["Dst IP"] == F.col("dst_orig"),
        "left"
    ).withColumn("Dst IP", F.coalesce(F.col("dst_synth"), F.col("Dst IP"))).drop("dst_orig", "dst_synth")
    
    return df_processed
