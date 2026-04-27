# 1. Pulisci la vecchia cache "sporca"
rm -rf preprocessed_cache/final_preprocessed_*.parquet
rm -rf models_cache/
rm -rf data/intel_cache/
# 2. Esegui la tua pipeline (prima run)
./run_pipeline.sh
cp results/centralized_results.csv results/tmp_run1.csv
# 3. Esegui di nuovo la tua pipeline (seconda run)
./run_pipeline.sh
cp results/centralized_results.csv results/tmp_run2.csv
# 4. Verifica che siano identici ignorando solo il tempo!
diff <(cut -d',' -f1-16,19- results/tmp_run1.csv) <(cut -d',' -f1-16,19- results/tmp_run2.csv)