#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Settings for the synth-cic-ids-2018 data generator."""

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parent

# Ingestion and day selection
DAYS = [
        # "Wednesday-14-02-2018",     # FTP-BruteForce, SSH-BruteForce
        "Thursday-15-02-2018",    # DoS-GoldenEye, DoS-Slowloris
        "Friday-16-02-2018",      # DoS-SlowHTTPTest, DoS-Hulk
        # "Tuesday-20-02-2018",       # DDoS-LOIC-HTTP, DDoS-LOIC-UDP
        # "Wednesday-21-02-2018",   # DDoS-LOIC-UDP, DDoS-HOIC
        #"Thursday-22-02-2018",    # Web-BruteForce, Web-XSS, Web-SQLi
        #"Friday-23-02-2018",      # Web attacks (continua)
        # "Wednesday-28-02-2018",   # Infiltration
        #"Thursday-01-03-2018",    # Infiltration (continua)
        #"Friday-02-03-2018",      # Bot
]
FORCE_REDOWNLOAD = False

# Dataset source
DATASET_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip"
DATASET_DIR = PIPELINE_ROOT / "CSECICIDS2018_improved"

# Paths
CACHE_DIR = PIPELINE_ROOT / "preprocessed_cache"
# Threat Intelligence Configuration
# =======================================================================
# FEED TEST SUMMARY (Apr 2026):
# The active feeds combine to generate a dynamic pool of unique 
# strictly malicious IPv4 addresses used dynamically to shape the attacks.
# =======================================================================
THREAT_INTEL_FEEDS = {
    # 1. ABUSEIPDB BLOCKLIST (ACTIVE MODE)
    # Category: Worst offenders of the internet, updated daily from AbuseIPDB.
    # Features ~170,000+ highly verified malicious IPs active in the last 30 days.
    "abuseipdb_30d": {
        "url": "https://raw.githubusercontent.com/borestad/blocklist-abuseipdb/main/abuseipdb-s100-30d.ipv4",
        "enabled": True,
    }
}

# =======================================================================
# Benign Data Configuration (Whitelists)
# Fetching IPs from the Borestad iplists repository to be used as trusted 
# endpoints for benign traffic mapping.
# =======================================================================
BENIGN_INTEL_FEEDS = {
    "googlebot": {
        "url": "https://raw.githubusercontent.com/borestad/iplists/refs/heads/main/google/googlebot.ipv4",
        "enabled": True,
    },
    "bingbot": {
        "url": "https://raw.githubusercontent.com/borestad/iplists/refs/heads/main/bing/bingbot.ipv4",
        "enabled": True,
    },
    "apple": {
        "url": "https://raw.githubusercontent.com/borestad/iplists/refs/heads/main/apple/apple.ipv4",
        "enabled": True,
    },
    "office365": {
        "url": "https://raw.githubusercontent.com/borestad/iplists/refs/heads/main/email/office365-exchange-smtp.ipv4",
        "enabled": True,
    }
}

# =======================================================================
# Custom Static IP Pools
# These arrays allow you to manually inject specific IPs into the pipeline
# without relying entirely on external feeds.
# =======================================================================

BASE_MALICIOUS_IPS = [
    # Add your own known-bad test IPs here. They will be merged with the intel feeds.
    "198.51.100.100", 
    "203.0.113.50", 
    "192.0.2.200", 
]

BASE_GOOD_PUBLIC_IPS = [
    # Verified safe infrastructure used to simulate normal public internet traffic
    "8.8.8.8", "8.8.4.4", "1.1.1.1", "1.0.0.1", 
    "9.9.9.9", "149.112.112.112", "208.67.222.222", "208.67.220.220",
    "142.250.0.0", "104.16.0.0", "151.101.1.69"
]

# Output settings
CACHE_ENABLED = True
SAMPLING_ENABLED = False
SAMPLE_SIZE = 500000
