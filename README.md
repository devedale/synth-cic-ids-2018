# nnids_pipeline

Pipeline autonoma per il processing del dataset CIC-IDS2018:

- **Ingestion**: scarica gli archivi PCAP giornalieri da AWS S3 (bucket pubblico `cse-cic-ids2018`)
- **Flow extraction**: genera flow CSV tramite CICFlowMeter (richiede Java 8)
- **Labeling**: assegna la label di attacco riga per riga via Apache Spark, basandosi su `configs/attack_schedule.yaml`
- **Preprocessing**: normalizzazione, feature selection, campionamento opzionale
- **Cache**: i risultati per giorno vengono salvati in `preprocessed_cache/<day>/`

## Struttura

```
nnids_pipeline/
├── main.py                        # entry-point CLI
├── setup.sh                       # setup automatico ambiente
├── requirements.txt
├── configs/
│   ├── settings.py                # percorsi, giorni, parametri
│   └── attack_schedule.yaml       # finestre di attacco per giorno
├── core/
│   ├── ingestion.py               # download S3 + estrazione + CICFlowMeter
│   ├── labeling.py                # MAP-REDUCE Spark: assegna Label
│   └── preprocessing.py          # normalizzazione e feature engineering
├── data/                          # archivi e PCAP (gitignored)
└── preprocessed_cache/            # output CSV (gitignored)
```

## Setup rapido

Il modo più semplice per partire da zero è clonare il repo ed eseguire lo script di setup:

```bash
git clone https://github.com/devedale/cic-ids-2018-labeling.git
cd cic-ids-2018-labeling/nnids_pipeline
bash setup.sh
source .venv/bin/activate
```

`setup.sh` si occupa di:
1. Installare le dipendenze di sistema (`openjdk-8-jdk`, `openjdk-17-jdk`, `unrar`, `p7zip-full`)
2. Creare il virtual environment `.venv` (se non esiste già)
3. Installare i pacchetti Python da `requirements.txt`
4. Eseguire uno smoke-test PySpark per verificare che tutto funzioni

> **Requisiti**: Linux, Python ≥ 3.8, `sudo` disponibile.

## Esecuzione

```bash
python main.py
```

Con giorni specifici:

```bash
python main.py --days Thursday-15-02-2018 Friday-16-02-2018
```

Forza rigenerazione ignorando la cache:

```bash
python main.py --force
```

Abilita salvataggio del DataFrame preprocessato:

```bash
python main.py --cache
```

### Argomenti CLI

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--days` | da `settings.py` | giorni da processare |
| `--force` | `False` | riscarta la cache e rigenera tutto |
| `--sample` | `20000` | dimensione campione (0 = tutti) |
| `--cache` / `--no-cache` | `False` | salva `preprocessed.csv` in cache |

## Configurazione giorni

I giorni attivi si configurano in `configs/settings.py` (lista `DAYS`).
Il dataset copre le seguenti giornate del CIC-IDS2018:

| Giorno | Attacchi |
|--------|----------|
| Wednesday-14-02-2018 | FTP-BruteForce, SSH-BruteForce |
| Thursday-15-02-2018 | DoS-GoldenEye, DoS-Slowloris |
| Friday-16-02-2018 | DoS-SlowHTTPTest, DoS-Hulk |
| Tuesday-20-02-2018 | DDoS-LOIC-HTTP, DDoS-LOIC-UDP |
| Wednesday-21-02-2018 | DDoS-LOIC-UDP, DDoS-HOIC |
| Thursday-22-02-2018 | Web-BruteForce, Web-XSS, Web-SQLi |
| Friday-23-02-2018 | Web attacks (continua) |
| Wednesday-28-02-2018 | Infiltration |
| Thursday-01-03-2018 | Infiltration (continua) |
| Friday-02-03-2018 | Bot |

## Output

Per ogni giorno processato vengono prodotti:

- `preprocessed_cache/<day>/benign_records.csv` — flow benigni con label
- `preprocessed_cache/<day>/attack_records.csv` — flow di attacco con label
- `preprocessed_cache/<day>/preprocessed.csv` — DataFrame preprocessato (solo con `--cache`)

## Notebook

Il file `run_pipeline.ipynb` permette di eseguire l'intera pipeline direttamente
da Jupyter o Google Colab senza nessuna configurazione manuale aggiuntiva.
