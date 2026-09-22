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

Built on [NVIDIA Cosmos-Framework](https://github.com/nvidia-cosmos/cosmos-framework).

**Paper:** [arXiv](https://arxiv.org/abs/2609.22762) &nbsp;&nbsp; **Project page:** [ffengc.github.io/drivereferee](https://ffengc.github.io/drivereferee/)

> **Code coming soon.**

## News & Updates

- [2026-09-19] Paper released on [arXiv](https://arxiv.org/abs/2609.22762).
- [2026-09-15] Submitted to ICRA 2027.

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
