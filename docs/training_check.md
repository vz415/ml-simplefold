# Check an existing fine-tuning run

`scripts/check_training_run.py` turns the existing collection, DockQ, MolProbity,
audit and report tools into one resumable workflow. It evaluates saved predictions;
it never launches new sampling, training or refinement.

The code is on branch `experiments/receptor-core-finetune`, alongside the training
and geometry tools. Activate the local SimpleFold environment. From the main
project directory in the current two-checkout setup:

```bash
python ../ml-simplefold-receptor-core-train/scripts/check_training_run.py run_id=58325408
```

From a checkout containing the training tools, use `python scripts/check_training_run.py`.
By default local artifacts are beneath the directory where the command is started.
Set `SIMPLEFOLD_LOCAL_ARTIFACT_DIR` or override `artifact_dir` to use another location.
Hydra settings are in `configs/training_check.yaml`; every invocation saves its
resolved configuration and stage receipt in the output directory.

## What it does

1. Connect directly with `ssh hpc3`, then use `hpc3y` for account-specific commands.
   Collect only structure outputs and text/configuration receipts through the RCIC
   transfer endpoint. Checksums identify missing or changed files; verified files
   are not transferred again. Completed local predictions and references are
   verified against their receipts and reused without contacting `/pub` at all.
   Set `refresh_downloads=true` to refresh all remote output receipts too.
   Weights and trajectory tensors are excluded.
2. Read the run's pinned `synthetic-selection.json`, prediction receipts and CAMEO
   references. Verify hashes, complete paired sample/seed sets, chain sequences
   and finite complex coordinates. Synthetic target files are resolved locally
   from their recorded paths or downloaded individually from the recorded dataset.
3. Run the existing four-receptor DockQ comparison, using a coherent chain mapping
   against the complete ligand dimer. Cache each result by prediction hash, target
   hash, chain identities, scorer source and DockQ version. Record native-contact
   recovery, interface RMSD, docking displacement, mean and weakest DockQ.
4. Reuse completed MolProbity results only when every input identity and the
   scientific method hash match. Otherwise stage the small verified PDB input set
   in home storage and submit the existing approved `hpc_sample.slurm
   --validate-geometry` CPU path. Durable outputs remain under `/pub`. The installed
   `simplefold-geometry` environment is reused. Source must be committed and pushed;
   the clean remote training checkout is updated with a fast-forward only.
5. Save the validation job ID immediately. Repeating the command resumes that job;
   a failed job is reported rather than silently resubmitted. The default wait is
   30 minutes, with progress every 20 seconds; if exceeded, rerun the same command.
6. Download and verify geometry receipts, independently reproduce coordinate
   overlap metrics, verify native clashscore normalization and unchanged heavy
   atoms, and check that incidental covalent linking and side-chain flips are off.
7. Produce Markdown, CSV, JSON and HTML reports, including per-target inspection
   and target-paired confidence intervals. Historical CAMEO columns are added only
   when references, seeds and original model outputs match exactly. Existing
   complex viewers are untouched. Append a deduplicated completion note to the
   configured Obsidian experiment directory.

## Useful settings

```bash
# All outputs are already local; no network, submission or GPU use.
python ../ml-simplefold-receptor-core-train/scripts/check_training_run.py \
  run_id=58325408 offline=true 'label=Cropped mixed'

# A specific existing geometry job; normally discovered automatically.
python ../ml-simplefold-receptor-core-train/scripts/check_training_run.py \
  run_id=58325408 geometry.job_id=58332178

# Do not write an Obsidian receipt.
python ../ml-simplefold-receptor-core-train/scripts/check_training_run.py \
  run_id=58325408 notes.enabled=false
```

Local outputs: `artifacts/analysis/training-checks/RUN_ID/`:

- `index.html`, `report.md`, `metrics.csv`, `summary.json`: complete comparison.
- `geometry/index.html`: individual target/sample selection.
- `geometry-summary.json`: geometry means and target-paired confidence intervals.
- `geometry-audit.json`: analysis integrity checks.
- `input-manifest.json`, `resolved-config.yaml`, `receipt.json`: provenance/resume state.

Only one compact geometry metrics file is copied into the report; the large raw
native diagnostics stay in `artifacts/remote-runs/GEOMETRY_JOB/`. Hash-keyed caches
are shared between runs. An offline check requires existing native geometry
results; it does not substitute approximate checks and claim completion.

The first supported input format is the `train_mixed.py` run layout: two conditions
(`pretrained`, `fine_tuned`), `retention/` with a completed CAMEO benchmark, and a
pinned synthetic selection. A legacy run can supply `selection=/path/to/selection.json`.
Complex targets remain six-chain ligand-dimer/type-I/type-II assemblies (A–F), as
required by the existing scorer. The script collects the five already-computed
OpenStructure CAMEO scores; it does not resample missing benchmark predictions.

## Interpretation

Synthetic complex scores measure training-target reconstruction, not independent
experimental validation. Cropped and full-length complex benchmarks have different
sequences; their raw counts must not be treated as a controlled cropping comparison.
Packing, local covalent geometry and general folding retention are reported
separately. Three samples from a complex are not three independent targets.
