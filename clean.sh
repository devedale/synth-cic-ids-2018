#!/bin/bash
echo "🧹 Cleaning up temporary directories and generated files..."

rm -rf spark_tmp/ 
rm -rf *.egg-info/ 
rm -rf models_cache/ 
rm -rf preprocessed_cache/ 
rm -rf results/
rm -rf data/intel_cache/
rm -rf data/visuals/

echo "✅ Cleanup complete!"
