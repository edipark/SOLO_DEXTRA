# SOLO_DEXTRA

Humanoid locomotion for the SOLO_DEXTRA robot, trained in Isaac Lab with Adversarial Motion Priors (AMP)
and deployed on a Raspberry Pi + Dynamixel AX-18A hardware stack.

Repository: <https://github.com/edipark/SOLO_DEXTRA>

This repository is a stripped-down fork of [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
that keeps only the framework code required to train, evaluate, and deploy the SOLO_DEXTRA
policy. Other tasks, demos, tutorials, and training frameworks from upstream Isaac Lab have
been removed.

## Layout

```
source/                                 # Isaac Lab framework (trimmed)
├── isaaclab/                           # core simulation / asset / MDP API
├── isaaclab_assets/                    # robot & primitive assets
├── isaaclab_contrib/                   # contrib sensors (required by core)
├── isaaclab_rl/                        # RL wrappers (SKRL etc.)
└── isaaclab_tasks/
    └── isaaclab_tasks/direct/SOLO_DEXTRA/
        ├── dextra_amp_env.py           # DirectRLEnv implementation
        ├── dextra_amp_env_cfg.py       # env / robot / domain randomization config
        ├── dextra_robot_cfg.py         # URDF-based articulation config
        ├── actuators/ax18a.py          # AX-18A compliance + punch + damping model
        ├── agents/                     # SKRL config (skrl_amp_cfg.yaml)
        ├── motions/                    # AMP reference motion (.npz)
        ├── solo_models.py              # teacher / student / estimator networks
        ├── train_dagger.py             # DAgger student distillation
        ├── train_state_estimator.py    # privileged-state estimator training
        └── play_*, rollout_log.py, run_ablation.py, ...

scripts/reinforcement_learning/skrl/    # generic SKRL trainer/player
├── train.py
├── play.py
└── train_stiffness_curriculum.py       # SOLO_DEXTRA-specific curriculum

SOLO_ws/                                # hardware deployment workspace (Raspberry Pi)
├── deploy.py
├── export_to_onnx.py
├── config.yaml
├── hardware/dynamixel_interface.py
├── inference/onnx_policy.py, onnx_estimator.py
└── scripts/, utils/, models/

apps/                                   # Omniverse Kit launch configs
isaaclab.sh / isaaclab.bat              # launcher
pyproject.toml, environment.yml         # project & conda env
```

## Quick start

Install Isaac Sim 4.5 and set up dependencies as described in the upstream
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/). Only the
dependency setup is relevant — the project code itself lives in this repo.

```bash
# train
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Dextra-Amp-Walk-Direct-v0 --headless

# play
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
    --task Isaac-Dextra-Amp-Walk-Direct-v0 --checkpoint <path>

# DAgger student
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/train_dagger.py \
    --task Isaac-Dextra-Amp-Walk-Direct-v0 --headless
```

Registered task id: `Isaac-Dextra-Amp-Walk-Direct-v0`.

## Hardware deployment

See [SOLO_ws/README.md](SOLO_ws/README.md) for the Raspberry Pi + Dynamixel AX-18A
deployment workflow (ONNX export, real-time inference loop, sysid scripts).

## License & attribution

This project is based on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) by the
Isaac Lab Project Developers, licensed under BSD-3-Clause. See [LICENSE](LICENSE)
and [CONTRIBUTORS.md](CONTRIBUTORS.md) for upstream credits.

SOLO_DEXTRA additions are released under the same BSD-3-Clause license.
