#!/bin/bash
#SBATCH --job-name=winmal_v20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=winmal_v20_%j.log
set -euo pipefail
# Resolve the extracted package. When sbatch copies this launcher to its spool
# folder, submission from the extracted package restores the correct directory.
task_bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$task_bundle_dir/SHA256_manifest.json" && -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/SHA256_manifest.json" ]]; then
  task_bundle_dir="$SLURM_SUBMIT_DIR"
fi
task_python="${WINMAL_PYTHON:-python}"
"$task_python" "$task_bundle_dir/verify_bundle.py"
exec "$task_python" "$task_bundle_dir/combined_dataset_slurm_gpu_full_analysis_seed_42.py" \
  --data-dir "$task_bundle_dir/Dataset" \
  --disk-file "$task_bundle_dir/Dataset/disk_dataset_preprocessed_high_accuracy.xlsx" \
  --memory-file "$task_bundle_dir/Dataset/memory_dataset_preprocessed_high_accuracy.xlsx" \
  --network-file "$task_bundle_dir/Dataset/network_dataset_preprocessed_high_accuracy.xlsx" \
  --output-dir "${WINMAL_OUTPUT_DIR:-$task_bundle_dir/results/winmal_v20_full}" \
  "$@"
