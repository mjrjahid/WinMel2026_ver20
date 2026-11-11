WINMAL2026 V20: GAUSSIAN NB PRECISION FIX + EXACT UPLOADED INPUTS

The latest probability error was reproduced in Disk GaussianNB with float32
calculations. GaussianNB now uses float64 for fitting and prediction, with the
same probability-check tolerance and no post-hoc probability repairs. Read the
updated experiment guide for the numerical evidence and validation artifacts.

Earlier jobs read reduced *_preprocessed.xlsx tables that lack the source
text and/or originating-report hash. This package supplies the three workbooks
uploaded most recently in the conversation. Their bytes are unchanged. Filenames
inside Dataset remove only the download suffixes '(3)'/'(5)'. Do not replace these
files with the reduced tables from an older run. They are preprocessed source
cohorts, not reconstructed raw reports or a recovered full-family inventory.

1. UPLOAD AND EXTRACT THE COMPLETE ZIP ON YOUR SERVER
For example, from /nfs/fjahiduzzaman/thesis:
 unzip WinMal2026_v20_Ready_To_Run.zip
 cd WinMal2026_v20

Keep Dataset, verify_bundle.py, SHA256_manifest.json and the main .py together.
The launcher verifies hashes and pins the exact three workbook paths. It works
regardless of the current working directory, including paths containing spaces.
Use the Python environment where you installed the experiment dependencies.
Optionally select it with WINMAL_PYTHON=/absolute/path/to/python.

2. RUN THE INPUT AUDIT FIRST (no model fitting)
 bash run_winmal_v20.sh --audit-only --skip-xai --output-dir ./results/v20_input_audit

3. RUN THE FULL EXPERIMENT SUITE
Within an existing SLURM allocation:
 srun bash run_winmal_v20.sh
Or on an appropriate compute machine directly:
 bash run_winmal_v20.sh

To submit a NEW SLURM allocation, first change into WinMal2026_v20, then:
 sbatch run_winmal_v20.sh
The launcher recovers the package path from SLURM_SUBMIT_DIR when sbatch copies
it into its spool folder. Adjust SBATCH resources/partition/time to your cluster.
For submission from a different directory use an absolute path with --wrap:
 sbatch --job-name=winmal_v20 --cpus-per-task=8 --mem=32G --time=24:00:00 --output=winmal_v20_%j.log --wrap='bash /nfs/fjahiduzzaman/thesis/WinMal2026_v20/run_winmal_v20.sh'
This package does not submit a job until you run a submission command.

Defaults: seeds 42,123,350; requested 10 folds; fixed minimum 10 source records
and independently evaluable identity groups; the v18 research protocols remain.
Use --experiments primary --skip-xai to run only predictive baselines first.
Read WinMal2026_Experiment_Guide.txt for the complete experiment definitions.
Missing real time/external/review evidence is reported explicitly.

4. OUTPUTS AND RESTARTS
Default full output: WinMal2026_v20/results/winmal_v20_full
For another run, give a NEW --output-dir. Use --resume only with identical script,
options and input bytes for a run started with this package. Do not resume v18/v19 with v20.
An audit directory and a model-run directory must be different.

5. INPUT SCOPE
Disk: 3,443 supplied records / 25 families.
Memory: 2,014 supplied records / 23 families.
Network: 707 supplied records / 8 families.
The script reports retained counts after its documented checks; they may differ.
It cannot restore families or observations removed before these workbooks were
created. The Network result applies to this eight-family cohort. Broader claims
need broader original data. Source text/identifiers are not invented or recovered
from the reduced numeric tables.

6. CHECKS
 python verify_bundle.py
 python combined_dataset_slurm_gpu_full_analysis_seed_42.py --self-test

Dependencies (if missing): python -m pip install -r requirements.txt
XGBoost/CatBoost are optional imports; install them to run all seven classifier
families. SHAP rendering requires shap. Existing working cluster packages need
not be reinstalled merely because this input package is new.

PRE-RUN IDENTITY AUDIT ON THESE EXACT UPLOADS
Disk: 3,443 supplied rows; 3,434 retained rows / 24 families (uniform min10).
Memory: 2,014 supplied rows; 1,135 retained rows / 15 families. The other 879
parent_sha256 values are numbers, not 64-character SHA-256 hashes. They cannot
establish report-subject identity. They are recorded/excluded from the primary
grouped benchmark. Genuine originating-report hashes are needed to recover them.
Network: 707 supplied and retained rows / 8 families.
With these independent identity components, requested 10-fold evaluation is
feasible at 2 folds for Disk and 10 folds for Memory/Network across seeds
42/123/350. The script records actual folds; do not call the Disk run ten-fold.
No missing hash, class label, or original text value is reconstructed by guessing.

QUICK NUMERICAL CHECK BEFORE THE FULL RUN
 bash run_winmal_v20.sh --datasets disk --models 'Naive Bayes' --experiments primary --skip-xai --output-dir ./results/v20_nb_check
A successful check is separate from the full run and has its own output folder.
GaussianNB model_specification.json should say dense_input_dtype=float64.
probability_validation.csv records each fold; invalid values still cause failure
with the model/fold and raw diagnostic arrays preserved.
