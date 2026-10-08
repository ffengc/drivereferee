# DriveReferee: Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models

Fengcheng Yu<sup>1</sup>, Dhruv Parikh<sup>1</sup>, Junjie Ye<sup>1</sup>, Maulik Bhatt<sup>2</sup>, Thang Vu<sup>2</sup>, Igor Vasiljevic<sup>3</sup>, Vitor Guizilini<sup>3†</sup>, Yue Wang<sup>1†</sup>

<sup>1</sup>University of Southern California &nbsp;&nbsp;&middot;&nbsp;&nbsp; <sup>2</sup>Woven by Toyota &nbsp;&nbsp;&middot;&nbsp;&nbsp; <sup>3</sup>Toyota Research Institute

† Equal advising.

Generative world-action models jointly generate future video and vehicle actions, but their
action branch is trained by imitation and gets no closed-loop geometric verdict. DriveReferee
keeps the safety rule explicit: a learned geometry readout predicts the scene state (drivable
area and future vehicle occupancy) from camera observations, and an analytic referee executes
the collision and drivable-area checks on it. During training the referee scores self-sampled
trajectories on ground-truth state and the resulting preferences are distilled into the policy;
at deployment the same referee runs on the predicted state and selects a safer candidate when
needed. On the full NAVSIM navtest, DriveReferee reaches 92.02 PDMS with a single front camera
and no external training data.

Built on [NVIDIA Cosmos-Framework](docs/cosmos_framework_readme.md).

**Paper:** [arXiv](https://arxiv.org/abs/2609.22762) &nbsp;&nbsp; **Project page:** [ffengc.github.io/drivereferee](https://ffengc.github.io/drivereferee/)

## News & Updates

- [2026-09-19] Paper released on [arXiv](https://arxiv.org/abs/2609.22762).
- [2026-09-15] Submitted to ICRA 2027.

## Setup

1. Training environment (Cosmos-Framework):
   ```shell
   uv sync --all-extras --group=cu130-train
   source .venv/bin/activate && export LD_LIBRARY_PATH=
   ```
2. Two NavSim devkit environments (they need numpy 1.x, keep them separate from the training env):
   the v2 devkit (`autonomousvision/navsim`, used for data conversion, the referee and EPDMS)
   and the v1.1 devkit (official PDMS). Download `navsim_logs/` and `maps/` per the devkit docs.
3. Base checkpoint and VAE:
   ```shell
   python -m cosmos_framework.scripts.convert_model_to_dcp --checkpoint-path Cosmos3-Nano -o $BASE_DCP
   uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir $(dirname $WAN_VAE_PATH)
   ```
4. Edit the paths in `scripts/env.sh`.

## Pipeline

All scripts read `scripts/env.sh`. Steps 2, 4, 5, 6, 7 need GPUs.

| step | command | what it does |
| --- | --- | --- |
| 1 | `bash scripts/01_convert_data.sh` | NavSim -> GEAR (832x480, 4 history + current + 8 future frames) |
| 2 | `bash scripts/02_train_base.sh 48k`, then `64k`, `merge64k`, `80k`, `merge80k` | base policy, 80k steps total |
| 3 | `bash scripts/03_occupancy_gt.sh` | occupancy GT (drivable + vehicles) and GT trajectories / speeds |
| 4 | `bash scripts/04_selfplay_referee_pairs.sh` | K=5 self-sampling on 15k scenes, referee grading, winner/loser pairs |
| 5 | `bash scripts/05_train_preference.sh` | referee distillation (beta = 10, lr 2e-5, ~30 exposures per pair) |
| 6 | `bash scripts/06_map_generator.sh` | map generator on the frozen vision tower, predicted maps for navtest |
| 7 | `bash scripts/07_gated_inference.sh` | default trajectory, thin-ice subset, second candidate, gated selection (K=2) |
| 8 | `bash scripts/08_evaluate.sh --build-caches`, then `bash scripts/08_evaluate.sh $WORK/gated/distilled/selected distilled` | official PDMS (v1.1) and EPDMS |

To score the undistilled base or the distilled policy without gating, evaluate
`$WORK/gated/<name>/seed0` from step 7 (run it with the corresponding merged checkpoint).
`scripts/08_evaluate.sh <dir> <name> <base name>` also prints the paired per-scene delta
with a bootstrap confidence interval.

## Layout

- `cosmos_framework/` Cosmos-Framework with the NavSim additions: dataset
  (`data/vfm/action/datasets/navsim_*.py`), history-frame conditioning
  (`data/vfm/action/transforms.py`), preference loss (`model/vfm/mot/preference_pair_loss.py`,
  hook in `model/vfm/omni_mot_model.py`), experiments
  (`configs/base/experiment/action/posttrain_config/action_policy_navsim_nano.py`),
  slim checkpoint merge (`scripts/merge_navsim_slim_dcp.py`).
- `examples/` launcher and the two TOML recipes.
- `tools/` data conversion (`navsim2gear/`), occupancy GT, inference dump, referee, pair
  building, map generator, gated selection, scoring helpers.
- `scripts/` the pipeline drivers above.

## Citation

```bibtex
@article{yu2026drivereferee,
  title   = {DriveReferee: Geometric Safety Verdicts Need Not Be Learned
             for Driving World-Action Models},
  author  = {Yu, Fengcheng and Parikh, Dhruv and Ye, Junjie
             and Bhatt, Maulik and Vu, Thang and Vasiljevic, Igor
             and Guizilini, Vitor and Wang, Yue},
  journal = {arXiv preprint arXiv:2609.22762},
  year    = {2026}
}
```
