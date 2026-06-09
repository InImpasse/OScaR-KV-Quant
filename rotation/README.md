# Rotation scripts

Granite and Gemma calibration/eval wrappers. **Full usage, benchmarks, and official-vs-local comparison are in the root [README.md](../README.md).**

Quick pointers:

- Calibration: `granite-4.0-1b/save_qkv_granite.sh` → `compute_rotation.sh`
- Default Granite rotations: `granite-4.0-1b/GPQA/seq30000_prompt118_group128/rotations/`
- Eval flags: all `eval_*.sh` scripts accept `--help` (see `scripts/lib/eval_cli.sh`)
