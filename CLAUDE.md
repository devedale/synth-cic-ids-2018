# Project AI Guidelines

1. **Continuous Documentation (`README.md` & `CLAUDE.md`)**:
   - Every time a new feature is added, an architecture rule is changed, or the system workflow evolves, the AI agent MUST update the `README.md` proactively.
   - Do NOT ask for permission to update the README. Just do it in your sequence of commits.
   - The README must strictly stay very professional, utilizing Mermaid diagrams, markdown tables, and alert boxes to clearly define the Data Pipeline architecture.
   - If the project's dependency surface or setup changes, update the setup block.
   
2. **Context Retention**:
   - The AI must use this directory and the `README.md` as the definitive source of truth to resume operations. The user should NEVER be forced to "re-transmit" past achievements or architecture details manually. Look at the `README.md` for understanding the PySpark ingestion, the IP caching mechanism, and the MLlib data preprocessing flow.

3. **Code Rules**:
   - Data scales up to 40-50GB. Pandas is implicitly banned for large loops. PySpark MLlib (`StandardScaler`, `StringIndexer`, `SparkSession`) and `repartition(10)` writing out to `Parquet` is the standard.
   - Adhere strictly to Git isolation and commit hygiene (never push directly to `main` without establishing a branch first).
