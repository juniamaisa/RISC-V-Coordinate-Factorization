# Coordinate Factorization Shapes Learned Geometry in Machine-Code Representations

This repository contains the framework focused in-program experiment for the "NeurIPS Workshops on Symmetry and Geometry in Neural Representations" paper.

A reproducible framework for studying coordinate factorization, linear accessibility, and learned representation geometry in RISC-V machine code.

## Scientific scope

- strictly in-program;
- workload: `sha-test`;
- context size: `S=32` only;
- representations: `SC`, `4C`, `32C`;
- seeds: `11`, `37`, `71`, paired across every representation and control;
- CUDA is used automatically when available; CPU is used when CUDA is unavailable;
- no `S=16`, no cross-program evaluation, no cross-workload evaluation, no leave-one-program-out claims.

The paper claim supported by this framework is:

```text
same RISC-V execution-context sample
        -> information-equivalent SC/4C/32C coordinate factorizations
        -> compact neural encoder
        -> learned geometry, accessibility, alignment, and counterfactual response
```

The intended contribution is not a new CRA detector. The security task is a controlled discrete substrate for studying neural representation geometry.

## Methodological controls incorporated in V10

Framework incorporates the following controls in the main experimental methodology:

1. representation-independent execution-context samples;
2. summarized static + dynamic CFG reference;
3. CFG-valid/invalid counterfactual pair construction;
4. leakage-resistant grouped splitting;
5. structural fingerprints;
6. duplicate detection;
7. representation collision analysis;
8. representation-induced shortcut audit.

V10 deliberately excludes opcode-semantic representations, control-flow-enriched representations, context-length research questions, multi-architecture research questions, and random-split baselines.

## Experiment count

| Block | Configuration | Unique CNN trainings |
|---|---:|---:|
| Main representation | SC/4C/32C × 3 seeds × S=32 | 9 |
| Capacity control | SC/32C × 3 seeds × S=32 | 6 |
| Padding-free control | SC/4C/32C × 3 seeds × S=32 | 9 |
| Geometry/probes/CKA/counterfactuals/cost | derived from trained models | 0 |
| **Total** | | **24** |

## Required data layout

Place the data directory in the repository root:

```text
DATA/sha-test/
├── common_overlap_filtered_samples_v2_sha-test.pickle
├── gadgets_mnemonic_sha-test.pickle
└── ...
```

## Install

CPU-compatible environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests
```

CUDA-capable Windows environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The configs use:

```json
"device": "auto"
```

So the framework resolves to CUDA when available and CPU otherwise. Each trained model records `requested_device`, `resolved_device`, CUDA availability, CUDA device name, deterministic status, and convergence diagnostics in its metrics JSON.

## Recommended execution

PowerShell:

```powershell
python run_neurreps.py pipeline `
  --main-config configs/neurreps_sha_test_focused_main.json `
  --capacity-config configs/neurreps_sha_test_focused_capacity_s32.json `
  --padding-config configs/neurreps_sha_test_focused_padding_s32.json
```

Bash:

```bash
python run_neurreps.py pipeline \
  --main-config configs/neurreps_sha_test_focused_main.json \
  --capacity-config configs/neurreps_sha_test_focused_capacity_s32.json \
  --padding-config configs/neurreps_sha_test_focused_padding_s32.json
```

Expected directories:

```text
results/neurreps_sha_test_s32_main/
results/neurreps_sha_test_s32_capacity/
results/neurreps_sha_test_s32_padding_free/
results/neurreps_sha_test_s32_main/report/
```

## Validate a completed run

```powershell
python scripts\validate_final_run.py
```

A valid paper run must contain the main, capacity, padding-free, nuisance, collision, layerwise, counterfactual, runtime, and reproducibility artifacts. It must also show model convergence and record the resolved device.

## Clean only V10 results

Windows:

```powershell
.\scripts\clean_results_windows.ps1
```

Linux/macOS:

```bash
bash scripts/clean_results_linux.sh
```

These scripts remove only V10 result directories. They do not touch `DATA/`.
