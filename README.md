# DriveReferee: Geometric Safety Verdicts Need Not Be Learned for Driving World-Action Models

Fengcheng Yu, Jiawei Yang, Dhruv Parikh, Junjie Ye, Maulik Bhatt, Thang Vu, Igor Vasiljevic, Vitor Guizilini†, Yue Wang†

† Corresponding authors.

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

<!-- TODO: fill in the arXiv and project page links once they are live -->
**Paper:** arXiv (coming soon) &nbsp;&nbsp; **Project page:** https://ffengc.github.io/drivereferee/ &nbsp;&nbsp; **Code:** coming soon

## News & Updates

- [2026-09-15] Submitted to ICRA 2027.

## Citation

<!-- TODO: fill in the arXiv number once the preprint is live -->

```bibtex
@article{yu2026drivereferee,
  title   = {DriveReferee: Geometric Safety Verdicts Need Not Be Learned
             for Driving World-Action Models},
  author  = {Yu, Fengcheng and Yang, Jiawei and Parikh, Dhruv and Ye, Junjie
             and Bhatt, Maulik and Vu, Thang and Vasiljevic, Igor
             and Guizilini, Vitor and Wang, Yue},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
