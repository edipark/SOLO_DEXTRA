"""
SOLO: State Estimator Training
===============================
24차원 관절 각도 정보(encoder)로부터 19차원 privileged state를 추정하는
State Estimator를 학습한다. LSTM, TCN, MLP 중 선택 가능하며, DAgger 방식의
반복 학습을 지원한다.

Usage (from repo root)::

    # LSTM (default, window=50)
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/train_state_estimator.py \\
        --teacher_checkpoint <path> --est_type LSTM --window 50 --headless

    # TCN
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/train_state_estimator.py \\
        --teacher_checkpoint <path> --est_type TCN --window 50 --headless

    # MLP (no history)
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/train_state_estimator.py \\
        --teacher_checkpoint <path> --est_type MLP --headless

    # DAgger rounds
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/train_state_estimator.py \\
        --teacher_checkpoint <path> --est_type LSTM --dagger_rounds 10 --headless
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import sys
import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="SOLO State Estimator Training")
parser.add_argument("--teacher_checkpoint", type=str, default="logs/skrl/dextra_amp_walk/task+amp/checkpoints/best_agent.pt",
                    help="Path to SKRL AMP best_agent.pt")
parser.add_argument("--num_envs", type=int, default=256)

# Estimator type
parser.add_argument("--est_type", type=str, default="LSTM", choices=["LSTM", "TCN", "MLP"],
                    help="State estimator type (default: LSTM)")
parser.add_argument("--window", type=int, default=50,
                    help="History window length (ignored for MLP)")

# Architecture
parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size (LSTM/MLP)")
parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--tcn_channels", type=int, nargs="+", default=[64, 128, 128],
                    help="TCN channel sizes per layer")
parser.add_argument("--tcn_kernel", type=int, default=3, help="TCN kernel size")

# Training
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--epochs", type=int, default=50, help="Training epochs per round")
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--collect_steps", type=int, default=2000,
                    help="Env steps for data collection per round")
parser.add_argument("--noise_levels", type=float, nargs="+", default=[0.0, 0.01, 0.02],
                    help="Action noise levels for data augmentation")

# DAgger
parser.add_argument("--dagger_rounds", type=int, default=10,
                    help="Number of DAgger rounds (0 = initial training only)")
parser.add_argument("--dagger_est_ratio", type=float, default=0.8,
                    help="Initial ratio of estimator usage during DAgger collection")
parser.add_argument("--dagger_est_ratio_final", type=float, default=1.0,
                    help="Final ratio of estimator usage at last DAgger round")
parser.add_argument("--dagger_est_ratio_schedule", type=str, default="linear",
                    choices=["linear", "constant"],
                    help="Schedule for est_ratio: 'linear' (default, ramp from initial to final) "
                         "or 'constant' (use --dagger_est_ratio for all rounds)")
parser.add_argument("--dagger_extra_rounds", type=int, default=0,
                    help="Additional DAgger rounds at 100%% est_ratio after the ramp phase")
parser.add_argument("--max_dataset_size", type=int, default=500000,
                    help="Maximum dataset size (random subsample if exceeded)")

# Evaluation
parser.add_argument("--eval_episodes", type=int, default=200,
                    help="Episodes per evaluation")
parser.add_argument("--max_episode_steps", type=int, default=1000,
                    help="Max steps per episode before timeout")
parser.add_argument("--robust_eval_seed_offset", type=int, default=10000,
                    help="Seed offset for the fixed held-out DR evaluation")

# Seed
parser.add_argument("--seed", type=int, default=42)

# Output
parser.add_argument("--output_dir", type=str, default="logs/solo_estimator",
                    help="Output directory for checkpoints and logs")

# Environment
parser.add_argument("--task", type=str, default="Isaac-Dextra-Amp-Walk-Direct-v0")
parser.add_argument("--agent_cfg_entry_point", type=str, default="skrl_amp_cfg_entry_point")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401

from solo_models import (
    ENCODER_DIM, PRIV_DIM, OBS_DIM, ACTION_DIM,
    TeacherPolicy, SkrlAgentWrapper,
    DataCollector, Evaluator, EstimatorTrainer, ModelSaver,
    build_estimator, force_skrl_isaaclab_reset,
)


def _unwrap_direct_env(env):
    """Return the DirectRLEnv below Gym/SKRL wrappers."""
    current = env
    for _ in range(32):
        if hasattr(current, "event_manager") and hasattr(current, "scene"):
            return current
        next_env = getattr(current, "_env", None)
        if next_env is None:
            next_env = getattr(current, "unwrapped", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    raise RuntimeError("Unable to unwrap DirectRLEnv")


def _move_startup_events_to_dagger_mode(events_cfg) -> list[str]:
    """Prevent automatic startup DR and expose it as an explicit DAgger mode."""
    moved = []
    for name in dir(events_cfg):
        if name.startswith("_"):
            continue
        term = getattr(events_cfg, name)
        if getattr(term, "mode", None) == "startup":
            term.mode = "dagger_dr"
            moved.append(name)
    return moved


def _snapshot_nominal_dynamics(env):
    """Snapshot the clean environment before the first DAgger round."""
    core_env = _unwrap_direct_env(env)
    asset = core_env.scene["robot"]
    core_env._estimator_nominal_dynamics = {
        "materials": asset.root_physx_view.get_material_properties().clone(),
        "masses": asset.root_physx_view.get_masses().clone(),
        "inertias": asset.root_physx_view.get_inertias().clone(),
        "joint_armature": asset.data.joint_armature.clone(),
        "actuators": {
            name: {
                "damping": actuator.damping.clone(),
                "effort_limit": actuator.effort_limit.clone(),
                "effort_limit_sim": actuator.effort_limit_sim.clone(),
                "punch_torque": getattr(actuator, "_punch_torque", None).clone()
                if hasattr(actuator, "_punch_torque") else None,
            }
            for name, actuator in asset.actuators.items()
        },
        "joint_velocity_noise_cfg": getattr(core_env, "_joint_velocity_observation_noise_cfg", None),
    }


def _restore_nominal_dynamics(core_env):
    """Restore clean dynamics after a DAgger collection rollout."""
    asset = core_env.scene["robot"]
    snapshot = core_env._estimator_nominal_dynamics
    physx_env_ids = asset._ALL_INDICES.cpu()
    asset.root_physx_view.set_material_properties(snapshot["materials"], physx_env_ids)
    asset.root_physx_view.set_masses(snapshot["masses"], physx_env_ids)
    asset.root_physx_view.set_inertias(snapshot["inertias"], physx_env_ids)
    asset.write_joint_armature_to_sim(snapshot["joint_armature"])
    for name, state in snapshot["actuators"].items():
        actuator = asset.actuators[name]
        actuator.damping[:] = state["damping"]
        actuator.effort_limit[:] = state["effort_limit"]
        actuator.effort_limit_sim[:] = state["effort_limit_sim"]
        if state["punch_torque"] is not None:
            actuator._punch_torque[:] = state["punch_torque"]
        if actuator.is_implicit_model:
            asset.write_joint_damping_to_sim(
                actuator.damping,
                joint_ids=actuator.joint_indices,
            )
            asset.write_joint_effort_limit_to_sim(
                actuator.effort_limit_sim,
                joint_ids=actuator.joint_indices,
            )
    core_env._joint_velocity_observation_noise_cfg = snapshot["joint_velocity_noise_cfg"]
    if hasattr(core_env, "_ax18a_effort_limit_ratio"):
        core_env._ax18a_effort_limit_ratio.fill_(float(core_env.cfg.effort_limit_ratio))


@contextmanager
def dagger_domain_randomization(env):
    """Enable physical DR only while collecting a DAgger round."""
    core_env = _unwrap_direct_env(env)
    core_env.event_manager.apply(mode="dagger_dr")
    try:
        yield
    finally:
        _restore_nominal_dynamics(core_env)


def evaluate_clean_and_dr(env, evaluator, teacher, estimator, num_episodes,
                          seed, robust_seed_offset, use_mlp=False):
    """Evaluate one checkpoint in both clean and fixed held-out DR environments."""
    clean_result = evaluator.evaluate_with_estimator(
        env, teacher, estimator, num_episodes,
        seed=seed, use_mlp=use_mlp,
    )

    # Seed before applying the event terms so every round receives the same
    # per-environment DR samples. The evaluator also uses this seed for reset.
    robust_seed = seed + robust_seed_offset
    torch.manual_seed(robust_seed)
    np.random.seed(robust_seed)
    with dagger_domain_randomization(env):
        dr_result = evaluator.evaluate_with_estimator(
            env, teacher, estimator, num_episodes,
            seed=robust_seed, use_mlp=use_mlp,
        )

    return clean_result, dr_result


# ============================================================================
# MAIN
# ============================================================================

@hydra_task_config(args_cli.task, args_cli.agent_cfg_entry_point)
def main(env_cfg, experiment_cfg):
    requested_device = getattr(args_cli, "device", "cuda:0")
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = requested_device
        if ":" in requested_device:
            torch.cuda.set_device(int(requested_device.split(":")[-1]))
    else:
        device = "cpu"

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    dagger_dr_terms = _move_startup_events_to_dagger_mode(env_cfg.events)
    env_cfg.termination_min_vel_x = 0.0  # 속도 terminate 비활성화 (estimator warm-up 보호)
    env_cfg.vel_window_min_vx = 0.0      # windowed 속도 terminate도 비활성화

    # Seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    est_type = args_cli.est_type.upper()
    use_mlp = est_type == "MLP"
    window = 1 if use_mlp else args_cli.window

    print("\n" + "=" * 70)
    print("  SOLO: State Estimator Training")
    print("=" * 70)
    print(f"  Type: {est_type}, Window: {window}")
    print(f"  Hidden: {args_cli.hidden_size}, Layers: {args_cli.num_layers}")
    print(f"  DAgger Rounds: {args_cli.dagger_rounds}")
    print(f"  Collect Steps: {args_cli.collect_steps}, Epochs: {args_cli.epochs}")
    print(f"  Seed: {args_cli.seed}, Device: {device}")
    print("=" * 70)

    # ── Environment ──
    print("\n[Setup] Environment")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    _snapshot_nominal_dynamics(env)
    print(f"  DAgger-only DR terms: {', '.join(dagger_dr_terms)}")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    force_skrl_isaaclab_reset(env)
    obs, _ = env.reset()
    print(f"  Obs: {obs.shape[-1]}, Envs: {args_cli.num_envs}")

    # ── Teachers ──
    print("\n[Setup] Teachers")
    skrl_runner = Runner(env, experiment_cfg)
    skrl_runner.agent.load(args_cli.teacher_checkpoint)
    skrl_teacher = SkrlAgentWrapper(skrl_runner.agent)
    print("  ✓ skrl Teacher (data collection)")

    teacher = TeacherPolicy(OBS_DIM, ACTION_DIM, device=device)
    teacher.load_from_checkpoint(args_cli.teacher_checkpoint, device=device)
    print(f"  ✓ Teacher Policy ({device})")

    # ── Output directory ──
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # teacher experiment name: .../logs/skrl/dextra_amp_walk/{exp_name}/checkpoints/best_agent.pt
    teacher_exp = os.path.basename(os.path.dirname(os.path.dirname(args_cli.teacher_checkpoint)))
    run_name = f"{est_type}_w{window}_seed{args_cli.seed}_{teacher_exp}"
    run_dir = os.path.join(args_cli.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n[Output] {run_dir}")

    model_saver = ModelSaver(run_dir, args_cli.teacher_checkpoint)

    # ── Build estimator ──
    estimator = build_estimator(
        est_type, hidden_size=args_cli.hidden_size, num_layers=args_cli.num_layers,
        num_channels=tuple(args_cli.tcn_channels), kernel_size=args_cli.tcn_kernel,
        device=device,
    )
    trainer = EstimatorTrainer(estimator, device, use_mlp=use_mlp)

    collector = DataCollector(window, ENCODER_DIM, PRIV_DIM, device)
    evaluator = Evaluator(window, ENCODER_DIM, PRIV_DIM,
                          args_cli.max_episode_steps, device)

    # ── Save config ──
    config = {
        "est_type": est_type,
        "window": window,
        "hidden_size": args_cli.hidden_size,
        "num_layers": args_cli.num_layers,
        "tcn_channels": args_cli.tcn_channels,
        "tcn_kernel": args_cli.tcn_kernel,
        "lr": args_cli.lr,
        "epochs": args_cli.epochs,
        "collect_steps": args_cli.collect_steps,
        "dagger_rounds": args_cli.dagger_rounds,
        "dagger_extra_rounds": args_cli.dagger_extra_rounds,
        "noise_levels": args_cli.noise_levels,
        "eval_episodes": args_cli.eval_episodes,
        "max_episode_steps": args_cli.max_episode_steps,
        "robust_eval_seed_offset": args_cli.robust_eval_seed_offset,
        "seed": args_cli.seed,
        "teacher_checkpoint": args_cli.teacher_checkpoint,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ==================================================================
    # Phase 1: 초기 데이터 수집 (Teacher + GT)
    # ==================================================================
    print("\n[Phase 1] Initial data collection with Teacher + GT")
    all_h, all_p, all_s = [], [], []
    for noise in args_cli.noise_levels:
        print(f"  noise={noise:.3f} ...", end=" ", flush=True)
        h, p, s, stats = collector.collect_with_teacher_gt(
            env, skrl_teacher, args_cli.collect_steps, noise=noise,
        )
        if h is not None:
            all_h.append(h)
            all_p.append(p)
            all_s.append(s)
            avg_ep = stats.get('avg_episode', float('nan'))
            std_ep = stats.get('std_episode', float('nan'))
            death = stats.get('death_rate', float('nan'))
            print(f"samples={stats['total_samples']:,}  avg_ep={avg_ep:.1f}±{std_ep:.1f}  death={death:.1f}%")
            if avg_ep < 100:
                print(f"  ⚠️  avg_ep < 100: teacher가 현재 env에서 정상 동작하지 않을 수 있음!")
        else:
            print("no data")

    histories = torch.cat(all_h)
    privileged = torch.cat(all_p)
    single_frames = torch.cat(all_s)
    print(f"  Total samples: {len(histories):,}")

    # Teacher GT 베이스라인 — env 설정이 올바른지 확인하는 가장 중요한 체크
    print("\n  [Teacher GT eval] teacher + GT priv로 평가 (estimator 없음) ...")
    gt_result = evaluator.evaluate_teacher_gt(
        env, teacher, args_cli.eval_episodes, seed=args_cli.seed,
    )
    print(f"  [Teacher GT eval] ep={gt_result['avg_episode']:.1f}±{gt_result['std_episode']:.1f}  "
          f"death={gt_result['death_rate']:.1f}%  timeout={gt_result['timeout_rate']:.1f}%")
    if gt_result['avg_episode'] < 100:
        print("  ⚠️  Teacher GT ep < 100 → stiffness/motion/checkpoint 불일치 의심. 계속 진행하면 의미없는 결과가 나올 수 있음.")

    # ==================================================================
    # Phase 2: 초기 학습
    # ==================================================================
    print("\n[Phase 2] Initial training")
    trainer.train(
        histories, privileged, single_frames,
        epochs=args_cli.epochs, lr=args_cli.lr, batch_size=args_cli.batch_size,
    )

    # 초기 clean / held-out DR 평가
    clean_result, dr_result = evaluate_clean_and_dr(
        env, evaluator, teacher, estimator, args_cli.eval_episodes,
        seed=args_cli.seed,
        robust_seed_offset=args_cli.robust_eval_seed_offset,
        use_mlp=use_mlp,
    )
    print(f"  Round 0 Clean: episode={clean_result['avg_episode']:.1f}, "
          f"death={clean_result['death_rate']:.1f}%, "
          f"timeout={clean_result['timeout_rate']:.1f}%")
    print(f"  Round 0 DR:    episode={dr_result['avg_episode']:.1f}, "
          f"death={dr_result['death_rate']:.1f}%, "
          f"timeout={dr_result['timeout_rate']:.1f}%")

    model_saver.save_estimator(estimator, est_type, args_cli.seed, 0, window,
                               {"avg_episode": clean_result["avg_episode"],
                                "death_rate": clean_result["death_rate"],
                                "clean_avg_episode": clean_result["avg_episode"],
                                "clean_death_rate": clean_result["death_rate"],
                                "dr_avg_episode": dr_result["avg_episode"],
                                "dr_death_rate": dr_result["death_rate"],
                                "pareto_best": True})

    training_log = [{
        "round": 0,
        **clean_result,
        "clean": clean_result,
        "dr": dr_result,
        "pareto_best": True,
    }]
    best_clean_episode = clean_result["avg_episode"]
    best_dr_episode = dr_result["avg_episode"]
    best_round = 0
    best_state = {k: v.cpu().clone() for k, v in estimator.state_dict().items()}

    # ==================================================================
    # Phase 3: DAgger Rounds
    # ==================================================================
    if args_cli.dagger_rounds > 0:
        total_rounds = args_cli.dagger_rounds + args_cli.dagger_extra_rounds
        print(f"\n[Phase 3] DAgger ({args_cli.dagger_rounds} ramp + {args_cli.dagger_extra_rounds} extra = {total_rounds} rounds)")

        for rd in range(1, total_rounds + 1):
            print(f"\n  ── Round {rd}/{total_rounds} ──")

            # est_ratio 결정: ramp phase → extra phase (100%)
            if rd <= args_cli.dagger_rounds:
                # Ramp phase
                if args_cli.dagger_est_ratio_schedule == "constant":
                    est_ratio = args_cli.dagger_est_ratio
                else:  # linear
                    if args_cli.dagger_rounds > 1:
                        est_ratio = args_cli.dagger_est_ratio + \
                            (args_cli.dagger_est_ratio_final - args_cli.dagger_est_ratio) * \
                            (rd - 1) / (args_cli.dagger_rounds - 1)
                    else:
                        est_ratio = args_cli.dagger_est_ratio_final
            else:
                # Extra phase: always 100%
                est_ratio = 1.0

            # 새 데이터 수집 (estimator 사용)
            with dagger_domain_randomization(env):
                new_h, new_p, new_s, c_stats = collector.collect_with_estimator(
                    env, teacher, estimator, args_cli.collect_steps,
                    est_ratio=est_ratio, noise=0.01,
                    use_mlp=use_mlp,
                )

            if new_h is not None:
                histories = torch.cat([histories, new_h])
                privileged = torch.cat([privileged, new_p])
                single_frames = torch.cat([single_frames, new_s])

                # 데이터셋 크기 제한
                if len(histories) > args_cli.max_dataset_size:
                    idx = torch.randperm(len(histories))[:args_cli.max_dataset_size]
                    histories = histories[idx]
                    privileged = privileged[idx]
                    single_frames = single_frames[idx]

                collect_ep = c_stats.get('avg_episode', float('nan'))
                collect_death = c_stats.get('death_rate', 0.0)
                print(f"    Collect: avg_ep={collect_ep:.1f}  death={collect_death:.1f}%  "
                      f"est_usage={c_stats.get('est_usage', 0):.0%}  "
                      f"dataset={len(histories):,} (est_ratio={est_ratio:.2f})")

            # 재학습
            trainer.train(
                histories, privileged, single_frames,
                epochs=args_cli.epochs, lr=args_cli.lr * 0.5,
                batch_size=args_cli.batch_size, verbose=False,
            )

            # Clean과 고정된 held-out DR 조건에서 모두 평가
            clean_result, dr_result = evaluate_clean_and_dr(
                env, evaluator, teacher, estimator, args_cli.eval_episodes,
                seed=args_cli.seed,
                robust_seed_offset=args_cli.robust_eval_seed_offset,
                use_mlp=use_mlp,
            )
            print(f"    Clean: episode={clean_result['avg_episode']:.1f}, "
                  f"death={clean_result['death_rate']:.1f}%, "
                  f"timeout={clean_result['timeout_rate']:.1f}%")
            print(f"    DR:    episode={dr_result['avg_episode']:.1f}, "
                  f"death={dr_result['death_rate']:.1f}%, "
                  f"timeout={dr_result['timeout_rate']:.1f}%")

            clean_episode = clean_result["avg_episode"]
            dr_episode = dr_result["avg_episode"]
            clean_improved = clean_episode > best_clean_episode
            clean_not_worse = clean_episode >= best_clean_episode
            dr_improved = dr_episode > best_dr_episode
            dr_not_worse = dr_episode >= best_dr_episode
            pareto_updated = (
                (clean_improved and dr_not_worse)
                or (dr_improved and clean_not_worse)
            )

            if pareto_updated:
                best_clean_episode = clean_episode
                best_dr_episode = dr_episode
                best_round = rd
                best_state = {k: v.cpu().clone() for k, v in estimator.state_dict().items()}
                print(f"    ★ New Pareto best: clean={best_clean_episode:.1f}, "
                      f"DR={best_dr_episode:.1f}")

            model_saver.save_estimator(estimator, est_type, args_cli.seed, rd, window,
                                       {"avg_episode": clean_result["avg_episode"],
                                        "death_rate": clean_result["death_rate"],
                                        "clean_avg_episode": clean_result["avg_episode"],
                                        "clean_death_rate": clean_result["death_rate"],
                                        "dr_avg_episode": dr_result["avg_episode"],
                                        "dr_death_rate": dr_result["death_rate"],
                                        "pareto_best": pareto_updated})
            training_log.append({
                "round": rd,
                **clean_result,
                "clean": clean_result,
                "dr": dr_result,
                "pareto_best": pareto_updated,
            })

        # Restore best
        estimator.load_state_dict(best_state)
        estimator.to(device)

    # ==================================================================
    # Save final / best model
    # ==================================================================
    best_path = os.path.join(run_dir, "best_estimator.pt")
    torch.save({
        "estimator_state_dict": estimator.state_dict(),
        "estimator_config": estimator.get_config(),
        "window": window,
        # Keep best_episode for compatibility with existing play/deploy code.
        "best_episode": best_clean_episode,
        "best_clean_episode": best_clean_episode,
        "best_dr_episode": best_dr_episode,
        "best_round": best_round,
        "selection": "pareto_clean_dr",
        "seed": args_cli.seed,
        "teacher_checkpoint": args_cli.teacher_checkpoint,
    }, best_path)

    # Save training log
    log_path = os.path.join(run_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({"config": config, "rounds": training_log}, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print(f"  Pareto best round: {best_round}")
    print(f"  Clean / DR episode: {best_clean_episode:.1f} / {best_dr_episode:.1f}")
    print(f"  Best model: {best_path}")
    print(f"  Training log: {log_path}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
