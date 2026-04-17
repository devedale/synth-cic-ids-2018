<div align="center">
  <h1>CIC-IDS-2018 DDoS Enhanced Dataset Generator</h1>
  <p><h3>An autonomous, out-of-core pipeline for generating threat-intelligence-augmented synthetic PCAP datasets.</h3></p>
</div>

---

## 📌 Project Overview
This tool extracts the massive (~40-50GB uncompressed) **CIC-IDS-2018 improved dataset** and scales its preprocessing natively out-of-core using **PySpark MLlib**. The generator solves fundamental reproducibility and scalability issues:
1. **Zero OOM (Out-of-Memory)**: Bypasses Pandas' memory constraints by leveraging PySpark's parallel `Parquet` MapReduce mapping.
2. **Reproducible Threat Intel**: Fetches dynamic IPs from public Threat Intelligence repositories but caches them via `JSON` locally to guarantee stable downstream reproducibility across dataset generations.
3. **Pipelined ML Transform**: Standardizes data scaling and categorical encoding out-of-core immediately during ingestion.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[CIC-IDS-2018 Improved Zip] -->|Local Extraction| B[Raw CSV Chunks]
    
    subgraph Reproducibility Enclave
    T[Threat Intel GitHub Feeds] -->|JSON Cache| C[Malicious / Benign IP Pools]
    end
    
    subgraph PySpark Ingestion Engine
    B --> D{Spark MapReduce}
    C --> D
    D -->|Src/Dst IP Injection| E[Unified Parquet Database]
    end
    
    subgraph PySpark ML Preprocessing
    E -->|VectorAssembler| F[StringIndexer]
    F -->|StandardScaler| G[fully_scaled_preprocessed.parquet]
    end
```

---

## ⚙️ Core Components

| Component | Responsibility | Technical Stack |
| :--- | :--- | :--- |
| **`core/ingestion.py`** | Downloads dataset, handles unzipping. Reads raw daily CSVs using `SparkSession`, creates deterministic subnet IP pools, and injects Threat feed indicators (e.g. `198.51.100.1` for malicious classes). Repartitions into out-of-core `unified_records.parquet` outputs per day. | PySpark SQL, UDFs, Parquet |
| **`core/preprocessing.py`** | Connects to the ingestion parquets, builds a Native PySpark Pipeline. Samples `df.sample()`, encodes the `Label` target column with `StringIndexer`, merges numerics via `VectorAssembler`, and normalizes ranges via `StandardScaler`. | PySpark MLlib |
| **`configs/settings.py`** | Centralizes HTTP feeds configuration, local filesystem pointers, sample size settings, and threat intel thresholds. | Constants |
| **`main.py`** | Entry-point script orchestrating Python context switching sequentially from PySpark Ingestion to PySpark Preprocessing. | Orchestrator |

---

## 🚀 Usage Guide

### 1. Requirements & Setup
Because the system runs on **PySpark**, your local environment must have Java runtime enabled.

```bash
# 1. Install Java (Linux/Ubuntu)
sudo apt install default-jre

# 2. Setup isolated environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core Python dependencies
pip install -r requirements.txt
```

### 2. Execution
Run the system via the entry script. (By default, downloading the 5.3GB ZIP is automatic if `CSECICIDS2018_improved/` does not exist).
```bash
python main.py
```

Optional CLI overrides:
- `--days`: Array of days to process (e.g. `--days Monday-12-02-2018`)
- `--force`: Ignore cached IP feeds and local parquets, rewriting everything.
- `--sample`: Set integer to downsample the final table (e.g. `--sample 500000`)
- `--no-cache`: Prevents saving the final Parquet back to the system disk.

---

> [!NOTE]
> **Data Integrity and Caching Pattern**
> 
> Threat Intelligence repositories block frequent scraping. To maintain reproducibility during ML modeling cycles, IP datasets are stored at `data/intel_cache/`. Delete `intel_cache` to force a new HTTP sweep and resample a totally different malicious injection topology.
