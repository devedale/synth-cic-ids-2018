#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal settings for the autonomous nnids_pipeline package."""

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parent

# Ingestion and day selection
DAYS = [
        # "Wednesday-14-02-2018",     # FTP-BruteForce, SSH-BruteForce
        # "Thursday-15-02-2018",    # DoS-GoldenEye, DoS-Slowloris
        # "Friday-16-02-2018",      # DoS-SlowHTTPTest, DoS-Hulk
        # "Tuesday-20-02-2018",       # DDoS-LOIC-HTTP, DDoS-LOIC-UDP
        # "Wednesday-21-02-2018",   # DDoS-LOIC-UDP, DDoS-HOIC
        #"Thursday-22-02-2018",    # Web-BruteForce, Web-XSS, Web-SQLi
        #"Friday-23-02-2018",      # Web attacks (continua)
        # "Wednesday-28-02-2018",   # Infiltration
        "Thursday-01-03-2018",    # Infiltration (continua)
        "Friday-02-03-2018",      # Bot

]
FORCE_REDOWNLOAD = False

# S3 source for CIC-IDS2018 archives
S3_BUCKET = "cse-cic-ids2018"
S3_REGION = "ca-central-1"
S3_PREFIX = "Original Network Traffic and Log data/"

# Tuesday archive is .rar, all the others are .zip
DAY_TO_ARCHIVE = {
	"Wednesday-14-02-2018": "pcap.zip",
	"Thursday-15-02-2018": "pcap.zip",
	"Friday-16-02-2018": "pcap.zip",
	"Tuesday-20-02-2018": "pcap.rar",
	"Wednesday-21-02-2018": "pcap.zip",
	"Thursday-22-02-2018": "pcap.zip",
	"Friday-23-02-2018": "pcap.zip",
	"Wednesday-28-02-2018": "pcap.zip",
	"Thursday-01-03-2018": "pcap.zip",
	"Friday-02-03-2018": "pcap.zip",
}

# Paths
ARCHIVES_DIR = PIPELINE_ROOT / "data" / "s3_archives"
PCAP_DIR = PIPELINE_ROOT / "data" / "pcaps"
FLOW_CSV_DIR = PIPELINE_ROOT / "data" / "cicflow_csv"
CACHE_DIR = PIPELINE_ROOT / "preprocessed_cache"

# CICFlowMeter installation in the main project root
CICFLOWMETER_ROOT = PROJECT_ROOT / "CICFlowMeter"

# Java settings required by jnetpcap
JAVA8_HOME = "/usr/lib/jvm/java-1.8.0-openjdk-amd64"
JAVA_XMX = "2g"

# Preprocessing
CACHE_ENABLED = False  # preprocessed.csv is never read, no need to save it
SAMPLE_SIZE = 20000

# Max rows per day kept during ingestion (applied before concat to cap RAM usage).
# Set to None to keep all rows (risk: OOM on large days).
INGEST_SAMPLE_SIZE = 500_000

# Labeling
ATTACK_SCHEDULE_YAML = PIPELINE_ROOT / "configs" / "attack_schedule.yaml"
