# BMP90 scoring and evaluation workflow

This is the entry point for BMP90 hypothesis development, sampling analysis,
and transfer between ligand–receptor complexes. Read it before BMP90 work.
The objective is to produce useful, evidence-supported candidate assemblies;
reference-free predictions must not be described as experimentally solved.

This guide does not authorize submissions, installations, or benchmarks.
Current user instructions control execution. On 2026-10-13 Vincent authorized
the five-target sequence below, with agent assistance and parallel A100/A30
experiments, followed by a review after BMP4. Earlier single-component pilot
plans and one-GPU limits are historical, not the current scientific strategy.

Vincent subsequently authorized paid A30 batches on `gpu` / `eehui_lab_gpu`
and proactive additional experiments within this five-target milestone. Keep
informative batches overlapping and give periodic recaps of findings, GPU-hours,
queue/runtime efficiency, and the next decisions. Do not make every target wait
for all results from its predecessor: a mapped, validated transfer hypothesis can
run provisionally while its parent is still being evaluated. Incorporate arriving
results into subsequent revisions and retain the exact parent evidence available
at submission. The first paid expansion is three tau jobs each for
BMP6–BMPR1A–ACVR2B and BMP7–BMPR1A–ACVR2A. Additional informative jobs are
authorized; no further routine permission question is needed within this scope.

## Initial five-target milestone and approved expansion

1. BMP6–ACVR1–ACVR2B: recover the interfaces covered by 9N4K.
2. BMP6–BMPR1A–ACVR2B: transfer the type-II module and evaluate against 9MIR.
3. BMP7–BMPR1A–ACVR2A: anchor the type-II interface with 1LX5 and test the
   transferred type-I hypothesis.
4. BMP7–ACVR1–ACVR2A: preserve a useful type-II module and test ACVR1 transfer.
5. BMP4–BMPR1A–ACVR2B: combine mapped successes with BMP4-specific evidence.

The initial milestone stopped after BMP4 for review with Vincent. On 2026-10-16,
after reviewing those results, Vincent approved the next bounded expansion:
BMP4–BMPR1A–ACVR2A, GDF5–BMPR1A–ACVR2A, and GDF5–BMPR1A–ACVR2B, in that
order. Each starts with a joint transferred recipe at tau 0.1, 0.3, and 0.8,
with fresh paired baseline/FK populations (nine sampling jobs). Favor A30s and
retain one sweep on A100. Submit this batch first, then launch restrained
Rosetta cleanup of BMP7–ACVR1–ACVR2A job58185779 FK sample2 and
BMP7–BMPR1A–ACVR2A job58139988 FK sample6 and baseline sample9. Keep the
baseline candidate labeled separately. This does not expand automatically to
the remaining inventory.

On 2026-10-17 Vincent requested a further iteration on those results: require
the missing type-II approach contact separately on each receptor copy for
BMP4–BMPR1A–ACVR2A and GDF5–BMPR1A–ACVR2B, and relax the selected recent
candidates. Wave18 contains six paired sampling jobs across the three agreed
tau values (three A30 and three A100; an 18 requested GPU-hour cap). Four
restrained CPU cleanups follow submission: BMP4 job58197150 FK3,
GDF5–ACVR2A job58197152 baseline8 and FK7, and GDF5–ACVR2B job58197156 FK0.
Keep baseline8 labeled unguided despite its byte-identical FK8 counterpart.
Relaxation evaluates local packing and covalent geometry; retaining a misplaced
receptor under restraints does not establish a correct interface.

Vincent approved wave19 on 2026-10-17: three A30 jobs each for
BMP4–BMPR1A–ACVR2A, BMP6–ACVR1–ACVR2B, and BMP7–ACVR1–ACVR2A at the
agreed tau values, with a total cap of 27 requested GPU-hours. Require the BMP4
type-II aromatic flanks and approach orientation together; require the BMP6 N60
loop and K78–Y490 wrist geometry together; and require the BMP7 Q89–L382 face
contact separately from its recovered T83 bridge. Preserve the other interface
modules and full constructs. The existing GDF5 A100 sweep remains queued.
Keep the previous best recipes while evaluating these revisions; no new targets
or relaxation jobs are included in this batch.

On 2026-10-18 Vincent requested another scoring iteration for BMP6 ACVR1
interactions and for BMP4 and BMP7, with all sampling submitted before GDF5
relaxation. Wave20 keeps the three-target, nine-A30-job tau sweep and a cap of
27 requested GPU-hours. Address the remaining BMP6 N60 approach while retaining
the wrist and type-II interfaces; improve BMP4 type-II placement and BMP7
type-II orientation alongside its improved Q89 contact. Then perform two
restrained CPU cleanups of GDF5–BMPR1A–ACVR2B job58209925 FK2 and baseline7.
The latter is unguided and byte-identical to FK7. Compare both candidates before
and after cleanup, retaining mutant-reference and inferred-interface limits.
This iteration introduces no new targets or automatic retries.

After reviewing all wave20 outcomes, Vincent requested the next batch on
2026-10-18 Pacific (2026-10-19 UTC). Wave21 contains nine A30 paired sampling
jobs at tau 0.1/0.3/0.8, with a cap of 27 requested GPU-hours and nine dependent
CPU evaluations. Build from retained successful modules and the recorded
failures: improve BMP4 second-copy type-II placement; improve BMP6 N60 geometry
while preserving wrist and type-II interfaces; and require BMP7 T83, Q89, and
type-II placement together. Verify that the proposed objectives distinguish
the actual previous defects and plausible alternative poses before sampling.
Retain full constructs and previous best recipes while results are pending.
This batch adds no targets, relaxation jobs, or automatic retries.

On 2026-10-19 Vincent requested the next BMP6 and BMP7 scoring refinements,
with their sampling submitted before restrained relaxation of BMP4. Wave22
contains six A30 paired sampling jobs across tau 0.1/0.3/0.8, capped at 18
requested GPU-hours, six dependent CPU evaluations, and then one CPU cleanup
of BMP4 v5 job58234682 FK5. Build BMP6 from its stronger retained N60/wrist
geometry; for BMP7 preserve the improved Q89/T83/type-II contacts and require
N60 separately so an average cannot hide its failure. BMP4 cleanup must compare
packing, covalent geometry, and both receptor copies before and after relaxation;
its partial homolog reference does not establish native BMP4 accuracy. Keep
full constructs, prior alternatives, and exact parent provenance. No new targets
or automatic retries are included.

For GDF5, the BMPR1A structure 3QB4 contains the R57A ligand mutation. Sample
the wild-type sequence, exclude the mutated residue from reference comparisons,
and avoid imposing its locally incompatible contact geometry. Label comparison
as mutant-surrogate agreement; masking the residue does not establish that the
surrounding wild-type conformation matches the mutant. Transferred type-II
interfaces remain hypotheses, with separate physical and evidence assessments.
The sequence starts with new targets; reproducing old BMP2 successes is not a
prerequisite. Reference qualifications below still apply, including receptor-trap
constructs and the species mapping needed for 1LX5.

The first wave per target uses one joint scoring hypothesis at tau 0.1, 0.3,
and 0.8, with a matched unguided population at each setting. Allocate distinct
seed triplets and record them before submission. Use available outcomes or an
explicit information-gathering rationale to choose independent restarts and
revisions; do not impose a full-wave completion barrier on useful parallel work.
A failed job is not an automatic reason to launch a duplicate. Keep one submission
owner and declare the job count and requested GPU-hours for each bounded batch.

The current campaign controller shares one lock between CPU registration and
submission. Wait for the registration job to complete and verify all planned
registration receipts before invoking submissions. Starting a submission while
registration holds this lock can outlast the local SSH timeout even when the
remote submission later succeeds. Reconcile the remote attempt, scheduler
response, submitted receipt, and remaining controller process before continuing;
never resubmit a slot whose successful job ID is already recorded.

Keep temporary results within each experiment checkout's
`artifacts/transfer-stage/runs/JOB_ID/` in HPC home storage, with canonical copies
on `/pub`. New sampling and evaluation stages share the temporary run; seal it
only after every writer completes. Use the checksum-verified retrieval and
cleanup procedure in [HPC operations](../hpc3y.md#retrieve-outputs). Remove only
verified temporary copies after local download and canonical publication, so
home storage stays bounded. Frozen historical jobs retain their original paths.

## Sources of truth

| Resource | What belongs there |
| --- | --- |
| [Scoring registry](../../configs/bmp90/scoring_registry.json) | Exact recipe snapshots, preferred and alternative recipes per complex, parent relationships, supporting runs, successful settings, and ranking history |
| [Interface-evidence registry](../../configs/bmp90/interface_evidence_registry.json) | Literature and coordinate evidence, generic chemistry rules, residue mappings, negative findings, provenance, and transfer limitations |
| [Failure registry](../../configs/bmp90/failure_registry.json) | Observed unsuccessful attempts, suspected causes, attempted fixes, follow-up results, and limits on generalizing a failure |
| [Evaluation protocol](../../configs/bmp90/evaluation_protocol.json) | Evidence allocation, reference-free calibration, physical assessment, acceptance records, and permitted claims |
| [Reference-gap audit](../../configs/bmp90/reference_gap_audit.json) | Dated experimental-reference searches, uncovered ligands, and construct/mutation qualifications; no-hit findings are not proof of absence |
| [Registry tooling](../../scripts/bmp90_registry.py) and local `artifacts/bmp90/` | Target inventory, immutable run manifests, append-only events, derived run index, and artifact references |
| [HPC operations](../hpc3y.md) and repository `AGENTS.md` | Access, storage, Slurm execution, and operational constraints |

The evaluation protocol and reference-gap audit in `configs/bmp90/` are standing
project instructions and reference material for the entire reconstruction process.
Follow their evidence-allocation, assessment, acceptance, and search-scope rules
alongside this workflow. Numeric acceptance thresholds remain pending calibration;
following the protocol does not make them validated. The three JSON
registries are research records, not automatic job launchers or a replacement
for resolved run configurations. Avoid creating a new YAML for every candidate:
keep recipe variants and rankings in the scoring registry, and record the exact
resolved configuration in each run's artifacts when launched.

## 1. Prioritize targets by interface coverage

Establish exact sequences, constructs, receptor identities, stoichiometry,
mutations, glycans, and chain mappings before interpreting reference coverage.
Record binding/signaling evidence separately from evidence about the pose.
An enumerated ligand–receptor combination is not automatically a stable complex.

Default priority is:

1. A compatible experimental full assembly.
2. Both exact ligand–receptor interfaces available in separate structures.
3. One exact ligand–receptor interface available.
4. Neither exact interface available.

### Reference-gap targets needing extra scrutiny

The [2026-10-13 reference-gap audit](../../configs/bmp90/reference_gap_audit.json)
found no direct signaling-receptor complex for **BMP3, BMP4, BMP5, BMP8A, BMP8B,
BMP15, GDF6, GDF7, and GDF10** under its experimental PDB accession,
polymer-description, and selected alias searches. Computed models were excluded.
Unpublished, unreleased, unindexed, or unusually named constructs may be missed:
this is a dated search finding, not proof that no structure exists.

Keep these targets in scope and attempt them using transferable scoring modules,
available biochemical evidence, and explicit interface hypotheses. Missing direct
references make validation harder, not the targets unworthy of study. Record exact
ligand–receptor combinations and coverage separately; this ligand-level list is
not an exhaustive pairwise evidence assessment. Unlisted biochemical pairings are
unassessed, not evidence-free or nonbinding.

Apply extra scrutiny to competing poses, independent-population recovery, and
sensitivity to uncertain restraints. Successful guidance alone cannot validate an
uncovered interface. Preserve qualifications for mutant BMP2/GDF5 references,
engineered and glycosylated BMP6 receptor traps, and ALK1/ALK6 interfaces outside
the working type-I panel. Reassess coverage if new evidence appears, retaining
the original search scope and date. See section 4 for evaluation consequences.

### Use covered interfaces as anchors

Within these groups, favor targets likely to teach rules transferable to several
others. Record coverage for each interface and receptor copy, including missing
regions. Distinguish exact references from partial, mutant, and homologous
surrogates. A target with no full assembly can still have excellent interface
references. Verify literature/PDB coverage before assigning the hardest group;
also distinguish targets with useful homologous templates from those without.

Recover and evaluate the known interface. Use that interface as an anchor while
sampling the remaining receptors. Report reference agreement and support for
newly inferred interfaces separately.

Separate reference structures can support both local docking arrangements, but
do not by themselves validate their combination. Check that both fit the same
ligand conformation, receptor copies occupy distinct sites, and the assembly
has plausible sterics, stoichiometry, symmetry, and membrane-facing geometry
where assessable. Use one coherent chain assignment; independent remapping must
not hide incompatible site occupancy. DockQ against guidance coordinates measures
reference recovery, not independent validation. Surrogate DockQ measures agreement
with that surrogate only.

## 2. Form a transferable scoring hypothesis

Consult successes, interface evidence, and failures before designing a new rule.
Search shared receptors and local interface similarity, then verify residue
identities, substitutions, species, constructs, and structural context. Shared
receptors make a transfer worth testing; they do not prove its contacts transfer.

Start with a promising combined type-I/type-II recipe rather than automatically
testing one component at a time. Preserve successful interface modules and make
small, explicit changes to uncertain ones. Record parent recipe, target mappings,
evidence IDs, expected improvement, and a result that would argue against the
hypothesis. Combine modules from different recipes only after checking their
compatibility in one assembly.

Keep observed structural contacts, energetic mutation effects, predicted contacts,
homologous transfers, and generic chemistry distinct. Mutation effects do not
identify atom partners without further evidence. Unknown is not absent; a mutation
with no affinity effect is not proof of no physical contact. Allocate evidence to
guidance, tuning, ranking, held-out evaluation, or context. Evidence used to tune
or select is no longer held out.

Use the eight inspection-rule classes in the evidence registry to assess chemistry
and structural integrity. Record atom definitions, thresholds, software versions,
protonation, missing atoms, and glycan assumptions. These are candidate checks,
not universal energy weights. Avoid double counting interactions. Use bounded,
smooth guidance where appropriate and report individual contributions; a strong
interface must not hide a failing receptor copy. Expensive post-sampling analyses
need not run at every FK checkpoint. Do not assume named analysis tools are installed.

Prefer biologically justified untrimmed constructs. Transfer a scoring module
separately from the source recipe's truncations. Consider trimming only when
evidence identifies a region causing interference despite principled guidance,
and preserve a compatible comparison. Vincent reports an untrimmed 2H62-related
success; its exact run and construct remain to be matched. The documented trimmed
recipe shortens BMPR1A, not BMP2.

## 3. Validate and test the hypothesis

Before GPU work, check mappings, score direction, missing-data behavior, symmetry
assignment, and whether displaced, overlapping, or duplicated-site arrangements
can exploit the score. Establish useful behavior against reference-compatible
models and plausible incorrect alternatives, not only easy clash-heavy decoys.

For future authorized runs, the agreed tau sweep is **0.1, 0.3, 0.8** and FK
checkpoint schedule is **[0.1, 0.3, 0.6, 0.75, 0.9]**. Historical successful
2H62 runs used a different checkpoint schedule including 0.97; do not rewrite
those receipts or describe the new schedule as already validated.

Use separately initialized populations and paired controls where applicable.
Reuse baselines only when constructs and relevant sampling settings are compatible;
an unmatched baseline must be labeled as such. Record seeds, tau, model, steps,
particle count, batching, ancestry, exact code/configuration, resources, and outputs.
FK descendants are correlated, not independent replicates.

Once execution is authorized, run a bounded set of promising hypotheses in parallel
on available authorized A100/A30 capacity. On A30, use model batching of two while
retaining the full ten-particle pool. Keep one submission owner; agents can develop
disjoint scoring hypotheses and analyze results. Follow operational instructions
and write experiment receipts when jobs are submitted. Capacity utilization is
not a reason to submit redundant or scientifically uninformative jobs.

## 4. Diagnose and decide

Evaluate every type-I/type-II copy and the whole assembly. Separate guidance
satisfaction, reference agreement where available, independent evidence, physical
checks, and run-level robustness. Raw reward values across different scoring
definitions are not a ranking metric.

### Select recipes across noise levels

Vincent clarified on 2026-10-14 that the tau sweep searches for useful settings
for each scoring function. Prefer the recipe and settings that produce the best
overall evaluated structures across tau 0.1, 0.3, and 0.8. Consider per-interface
DockQ where a suitable reference exists, support for uncovered interfaces, and
whole-assembly physical quality together. The largest single DockQ or optimized
reward alone does not define the best overall structure.

A recipe need not improve at every tau. A weaker result at one noise level does
not disqualify it when another setting produces better overall candidates. Keep
all run outcomes, winning settings, seeds, ancestry, and reference qualifications
in the registry. Treat cross-tau consistency as supporting robustness information;
do not make uniform performance a prerequisite for choosing a useful recipe.
Sampling variability is a possible explanation for an isolated weaker result,
not an established cause. Do not spend experiments distinguishing contact-rule
effects from sampling difficulty by default. Preserve successful constraints and
address the remaining structural defects; investigate causes when a systematic
problem obstructs progress.

| Finding | Next action |
| --- | --- |
| Type I recovers well; type II fails | Preserve type-I rules and revise the type-II hypothesis |
| Local interfaces recover but assembly clashes | Inspect joint occupancy, ligand conformation, incompatible references, and competing constraints |
| An uncertain restraint conflicts with an otherwise plausible interface | Test a targeted weight change or removal, with controlled settings |
| A specific region repeatedly interferes | Investigate mapping, construct and structural integrity before a justified trimming comparison |
| Multiple distinct plausible arrangements remain | Retain alternatives and identify a discriminating observation |
| A hypothesis repeatedly fails without a new evidence-backed revision | Record the failure and deprioritize it within the finite experiment budget |
| A job fails operationally | Fix execution; do not label the scoring hypothesis biologically unsuccessful |

Use component ablations to answer a specific question, not as the default search
strategy. Recurrence across seeds and tau is robustness evidence, not a correctness
probability: a wrong restraint can repeatedly recover the same wrong pose.

For reference-free targets, follow the evaluation protocol. Calibrate thresholds
against reference-backed examples and realistic incorrect alternatives before
awarding calibrated acceptance labels. Until then, record assessments as provisional
with threshold status pending. Report unknown/unassessed explicitly. Never compute
an invented DockQ or let a mean hide the weakest interface. Support can differ by
interface and structural scale; retain unresolved alternatives.

For the [reference-gap targets](#reference-gap-targets-needing-extra-scrutiny),
apply this protocol to every uncovered interface, explicitly report uncertainty,
and retain competing plausible arrangements when independent evidence cannot
distinguish them. Prioritize restraint-sensitivity checks and independent-run
comparisons; neither robustness nor a high optimized reward substitutes for
independent support. Identify a discriminating contact or experiment where
possible. Do not claim DockQ or numerical confidence without a suitable basis.

## 5. Record outcomes and expand

Update the scoring registry with settings and run evidence, including the practical
preferred recipe, metric-specific leaders, alternatives, and any former best.
Preserve exact historical recipes and append ranking changes. Update the failure
registry with observed symptoms separately from causal hypotheses; preserve later
rescues. Add new contact observations to the evidence registry with their provenance
and scope. Predictions remain predictions or transferable hypotheses, never newly
invented experimental evidence.

Expand first to related interfaces whose mappings and chemistry support transfer,
then to harder combinations. Revisit earlier failures when genuinely new evidence
or conditions justify it. A reference-free outcome may be a useful candidate
ensemble rather than a uniquely supported pose. State what would discriminate it.

## Worked transfer example: BMP2 to BMP4

The preferred recorded BMP2–BMPR1A–ACVR2B starting recipe is trimmed L61-on,
supported by successful tau 0.3 and 0.8 runs. L61-off remains a supported alternative
with slightly higher reported ensemble DockQ in its control; this does not establish
universal superiority or necessity of L61. See the scoring registry for run IDs,
hardware differences, and mutant-reference limitations.

For BMP4–BMPR1A–ACVR2B, first map the BMP2-derived BMPR1A and ACVR2B modules
onto the actual BMP4 construct. Incorporate BMP4-specific mutation/binding evidence
without turning mutation effects into invented atom contacts. Use both interface
modules together, retain L61 only as an explicitly mapped optional hypothesis,
and start with untrimmed constructs unless evidence warrants otherwise. Test the
agreed noise levels and diagnose each interface independently. Report BMP2-template
similarity as transferred-reference agreement, not native BMP4 accuracy. Record
successes, failures, and revised hypotheses in the linked registries.

This example describes a proposed transfer, not a completed BMP4 experiment.

## Reference-file provenance

The repository copies were imported byte-for-byte from Vincent's supplied files
on 2026-10-13. Original delivery files and acknowledgements remain preserved as
transport receipts; routine work should use the repository paths above.

| Repository file | Original delivery directory | Imported SHA-256 |
| --- | --- | --- |
| `configs/bmp90/evaluation_protocol.json` | `2026-10-13_103822_bmp-complex-evaluation-protocol` | `2ffb1b0e6699bb4ebf68133c18fc848f163e6a32431990f5f15606d1f42a1ca1` |
| `configs/bmp90/reference_gap_audit.json` | `2026-10-13_105304_bmp-pdb-reference-gap-audit` | `982c76217314c927bd5328b83cfebd701802df7abd4367a54c2ed73bbdf1ff96` |

Original delivery directories are under
`/Users/vincent_zaballa/Development/boltzgen/handoffs/`. These hashes identify the
imported versions, not a requirement that future authorized revisions retain
the same bytes. Preserve provenance and record substantive revisions.
