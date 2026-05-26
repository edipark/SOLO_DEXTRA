"""
SOLO: Ablation Study & Evaluation
==================================
학습된 State Estimator를 평가하고, 여러 설정(LSTM/TCN/MLP, history length,
DAgger 유무 등)에 대한 ablation study를 수행한다. 복수 random seed를 지원하며,
통계적 결과(mean ± std)를 리포트한다.

Usage (from repo root)::

    # 이미 학습된 estimator들을 평가
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/run_ablation.py \\
        --teacher_checkpoint <path> \\
        --estimator_dirs logs/solo_estimator/LSTM_w50_seed42_* logs/solo_estimator/TCN_w50_seed42_* \\
        --headless

    # 또는 학습부터 전체 ablation 자동 실행 (train + eval)
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/run_ablation.py \\
        --teacher_checkpoint <path> --run_training --seeds 3 --headless

    # 빠른 테스트
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/run_ablation.py \\
        --teacher_checkpoint <path> --run_training --fast --headless
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import torch
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="SOLO Ablation Study & Evaluation")
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)

# Mode: evaluate pre-trained OR train+evaluate
parser.add_argument("--estimator_dirs", type=str, nargs="*", default=None,
                    help="Paths to pre-trained estimator directories (each with best_estimator.pt)")
parser.add_argument("--run_training", action="store_true",
                    help="Train all ablation configs from scratch before evaluating")

# Random seeds (for --run_training mode)
parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
parser.add_argument("--seed_start", type=int, default=42, help="Starting seed value")

# Evaluation settings
parser.add_argument("--eval_episodes", type=int, default=1000)
parser.add_argument("--max_episode_steps", type=int, default=1000)

# Training settings (for --run_training mode)
parser.add_argument("--dagger_rounds", type=int, default=10)
parser.add_argument("--dagger_extra_rounds", type=int, default=0,
                    help="Additional DAgger rounds at 100%% est_ratio after the ramp phase")
parser.add_argument("--collect_steps", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)

# Presets
parser.add_argument("--fast", action="store_true")
parser.add_argument("--thorough", action="store_true")

# Ablation options
parser.add_argument("--skip_history_ablation", action="store_true")
parser.add_argument("--skip_mlp", action="store_true")
parser.add_argument("--skip_tcn", action="store_true")

# Output
parser.add_argument("--output_dir", type=str, default="logs/solo_ablation")
parser.add_argument("--task", type=str, default="Isaac-Dextra-Amp-Walk-Direct-v0")
parser.add_argument("--agent_cfg_entry_point", type=str, default="skrl_amp_cfg_entry_point")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Presets
if args_cli.thorough:
    PRESET = {"collect": 5000, "epochs": 60, "noise": [0.0, 0.01, 0.02, 0.03]}
    PRESET_NAME = "THOROUGH"
elif args_cli.fast:
    PRESET = {"collect": 1000, "epochs": 20, "noise": [0.0, 0.02]}
    PRESET_NAME = "FAST"
else:
    PRESET = {"collect": 2000, "epochs": 40, "noise": [0.0, 0.01, 0.02]}
    PRESET_NAME = "DEFAULT"

COLLECT_STEPS = args_cli.collect_steps or PRESET["collect"]
EPOCHS = args_cli.epochs or PRESET["epochs"]
NOISE_LEVELS = PRESET["noise"]

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
    build_estimator, load_estimator, force_skrl_isaaclab_reset,
)


# ============================================================================
# ABLATION EXPERIMENT DEFINITIONS
# ============================================================================

def get_ablation_experiments(skip_history=False, skip_mlp=False, skip_tcn=False):
    """Ablation 실험 목록을 반환한다.

    Returns:
        list of (name, est_type, window, use_dagger) tuples
    """
    experiments = [
        # 제안 방법
        ("LSTM_DAgger_w50", "LSTM", 50, True),
        # DAgger 없이 (initial training only)
        ("LSTM_Initial_w50", "LSTM", 50, False),
    ]

    if not skip_tcn:
        experiments.append(("TCN_DAgger_w50", "TCN", 50, True))

    if not skip_mlp:
        experiments.append(("MLP_DAgger", "MLP", 1, True))

    if not skip_history:
        experiments.extend([
            ("LSTM_DAgger_w10", "LSTM", 10, True),
            ("LSTM_DAgger_w25", "LSTM", 25, True),
            ("LSTM_DAgger_w100", "LSTM", 100, True),
        ])

    return experiments


# ============================================================================
# TRAIN + EVALUATE (single experiment, single seed)
# ============================================================================

def run_single_experiment(env, teacher, skrl_teacher, device, config,
                          model_saver, exp_name, est_type, window,
                          use_dagger, seed):
    """단일 실험 (학습 + 평가) 실행."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    use_mlp = est_type == "MLP"

    print(f"\n{'=' * 60}")
    print(f"  {exp_name} (Seed {seed})")
    print(f"  Type: {est_type}, Window: {window}, DAgger: {use_dagger}")
    print(f"{'=' * 60}")

    collector = DataCollector(window, ENCODER_DIM, PRIV_DIM, device)
    evaluator = Evaluator(window, ENCODER_DIM, PRIV_DIM,
                          config["max_episode_steps"], device)

    estimator = build_estimator(est_type, hidden_size=config["hidden_size"],
                                num_layers=config["num_layers"], device=device)
    trainer = EstimatorTrainer(estimator, device, use_mlp=use_mlp)

    # ── 초기 데이터 수집 ──
    print("    [1] Collecting initial data...")
    all_h, all_p, all_s = [], [], []
    for noise in config["noise_levels"]:
        h, p, s, stats = collector.collect_with_teacher_gt(
            env, skrl_teacher, config["collect_steps"], noise=noise,
        )
        if h is not None:
            all_h.append(h)
            all_p.append(p)
            all_s.append(s)

    histories = torch.cat(all_h)
    privileged = torch.cat(all_p)
    single_frames = torch.cat(all_s)
    print(f"        Total samples: {len(histories):,}")

    # ── 초기 학습 ──
    print("    [2] Initial training...")
    trainer.train(histories, privileged, single_frames,
                  epochs=config["epochs"], verbose=True)

    # ── Round 0 평가 ──
    result = evaluator.evaluate_with_estimator(
        env, teacher, estimator, config["eval_episodes"],
        seed=seed, use_mlp=use_mlp,
    )
    print(f"    [3] Round 0: episode={result['avg_episode']:.1f}, "
          f"death={result['death_rate']:.1f}%, timeout={result['timeout_rate']:.1f}%")

    model_saver.save_estimator(estimator, exp_name, seed, 0, window,
                               {"avg_episode": result["avg_episode"],
                                "death_rate": result["death_rate"]})

    rounds_data = [{"round": 0, **result}]
    best_episode = result["avg_episode"]
    best_state = {k: v.cpu().clone() for k, v in estimator.state_dict().items()}

    # ── DAgger ──
    dagger_extra_rounds = config.get("dagger_extra_rounds", 0)
    if use_dagger and (config["dagger_rounds"] > 0 or dagger_extra_rounds > 0):
        total_rounds = config["dagger_rounds"] + dagger_extra_rounds
        print(f"    [4] DAgger ({config['dagger_rounds']} ramp + {dagger_extra_rounds} extra = {total_rounds} rounds)...")

        est_ratio_init = 0.8
        est_ratio_final = 1.0

        for rd in range(1, total_rounds + 1):
            # est_ratio 스케줄링: ramp phase → extra phase (100%)
            if rd <= config["dagger_rounds"]:
                if config["dagger_rounds"] > 1:
                    est_ratio = est_ratio_init + \
                        (est_ratio_final - est_ratio_init) * \
                        (rd - 1) / (config["dagger_rounds"] - 1)
                else:
                    est_ratio = est_ratio_final
            else:
                est_ratio = 1.0

            new_h, new_p, new_s, c_stats = collector.collect_with_estimator(
                env, teacher, estimator, config["collect_steps"],
                est_ratio=est_ratio, noise=0.01, use_mlp=use_mlp,
            )

            if new_h is not None:
                histories = torch.cat([histories, new_h])
                privileged = torch.cat([privileged, new_p])
                single_frames = torch.cat([single_frames, new_s])

                if len(histories) > 500000:
                    idx = torch.randperm(len(histories))[:500000]
                    histories, privileged, single_frames = (
                        histories[idx], privileged[idx], single_frames[idx])

            trainer.train(histories, privileged, single_frames,
                          epochs=config["epochs"], lr=config["lr"] * 0.5, verbose=False)

            result = evaluator.evaluate_with_estimator(
                env, teacher, estimator, config["eval_episodes"],
                seed=seed, use_mlp=use_mlp,
            )
            phase = "extra" if rd > config["dagger_rounds"] else "ramp"
            print(f"        Round {rd} [{phase}]: est_ratio={est_ratio:.2f}, "
                  f"episode={result['avg_episode']:.1f}, "
                  f"death={result['death_rate']:.1f}%, timeout={result['timeout_rate']:.1f}%")

            model_saver.save_estimator(estimator, exp_name, seed, rd, window,
                                       {"avg_episode": result["avg_episode"],
                                        "death_rate": result["death_rate"]})
            rounds_data.append({"round": rd, **result})

            if result["avg_episode"] > best_episode:
                best_episode = result["avg_episode"]
                best_state = {k: v.cpu().clone() for k, v in estimator.state_dict().items()}

        estimator.load_state_dict(best_state)
        estimator.to(device)
        model_saver.save_estimator(estimator, exp_name + "_best", seed, -1, window,
                                   {"best_episode": best_episode})

    return {
        "exp_name": exp_name,
        "seed": seed,
        "window": window,
        "est_type": est_type,
        "use_dagger": use_dagger,
        "rounds": rounds_data,
        "final": rounds_data[-1],
        "estimator_config": estimator.get_config(),
    }


# ============================================================================
# EVALUATE PRE-TRAINED MODELS
# ============================================================================

def evaluate_pretrained(env, teacher, device, estimator_dirs, eval_episodes,
                        max_episode_steps, seeds):
    """이미 학습된 estimator 체크포인트들을 평가한다."""
    results = {}

    for est_dir in estimator_dirs:
        ckpt_path = os.path.join(est_dir, "best_estimator.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {ckpt_path} not found")
            continue

        estimator, ckpt = load_estimator(ckpt_path, device=device)
        cfg = ckpt["estimator_config"]
        window = ckpt.get("window", 50)
        exp_name = os.path.basename(est_dir)
        use_mlp = cfg["type"] == "MLP"

        print(f"\n  Evaluating: {exp_name} ({cfg['type']}, w={window})")
        evaluator = Evaluator(window, ENCODER_DIM, PRIV_DIM,
                              max_episode_steps, device)

        seed_results = []
        for seed in seeds:
            result = evaluator.evaluate_with_estimator(
                env, teacher, estimator, eval_episodes,
                seed=seed, use_mlp=use_mlp,
            )
            seed_results.append(result)
            print(f"    Seed {seed}: episode={result['avg_episode']:.1f}, "
                  f"death={result['death_rate']:.1f}%")

        results[exp_name] = {
            "seed_results": seed_results,
            "config": cfg,
            "window": window,
        }

    return results


# ============================================================================
# AGGREGATION & REPORTING
# ============================================================================

def aggregate_results(all_seed_results):
    """여러 seed 결과를 집계한다."""
    aggregated = {}
    first_seed = list(all_seed_results.values())[0]

    for exp_name in first_seed.keys():
        exp_data = [sr[exp_name] for sr in all_seed_results.values() if exp_name in sr]

        if exp_name == "baseline":
            metrics = ["avg_episode", "std_episode", "death_rate", "timeout_rate"]
            aggregated[exp_name] = {}
            for m in metrics:
                vals = [d[m] for d in exp_data]
                aggregated[exp_name][f"{m}_mean"] = np.mean(vals)
                aggregated[exp_name][f"{m}_std"] = np.std(vals)
        else:
            aggregated[exp_name] = {
                "window": exp_data[0]["window"],
                "est_type": exp_data[0].get("est_type", "?"),
                "use_dagger": exp_data[0]["use_dagger"],
                "seeds": [d["seed"] for d in exp_data],
            }
            metrics = ["avg_episode", "std_episode", "death_rate", "timeout_rate"]
            for m in metrics:
                vals = [d["final"][m] for d in exp_data]
                aggregated[exp_name][f"{m}_mean"] = np.mean(vals)
                aggregated[exp_name][f"{m}_std"] = np.std(vals)

            if "rounds" in exp_data[0]:
                num_rounds = len(exp_data[0]["rounds"])
                aggregated[exp_name]["rounds"] = []
                for r in range(num_rounds):
                    rd = {}
                    for m in metrics:
                        vals = [d["rounds"][r][m] for d in exp_data if r < len(d["rounds"])]
                        if vals:
                            rd[f"{m}_mean"] = np.mean(vals)
                            rd[f"{m}_std"] = np.std(vals)
                    aggregated[exp_name]["rounds"].append(rd)

    return aggregated


def print_summary(aggregated):
    baseline_ep = aggregated["baseline"]["avg_episode_mean"]
    baseline_std = aggregated["baseline"]["avg_episode_std"]

    print(f"\n  {'Method':<30} {'Episode':<20} {'Death%':<15} {'Timeout%':<15} {'Ratio%':<10}")
    print("  " + "-" * 90)

    print(f"  {'Teacher (GT)':<30} "
          f"{baseline_ep:<8.1f}±{baseline_std:<10.1f} "
          f"{aggregated['baseline']['death_rate_mean']:<7.1f}±{aggregated['baseline']['death_rate_std']:<6.1f} "
          f"{aggregated['baseline']['timeout_rate_mean']:<7.1f}±{aggregated['baseline']['timeout_rate_std']:<6.1f} "
          f"{'100.0':<10}")

    for name, data in aggregated.items():
        if name == "baseline":
            continue
        ep_mean = data["avg_episode_mean"]
        ep_std = data["avg_episode_std"]
        ratio = ep_mean / baseline_ep * 100
        print(f"  {name:<30} "
              f"{ep_mean:<8.1f}±{ep_std:<10.1f} "
              f"{data['death_rate_mean']:<7.1f}±{data['death_rate_std']:<6.1f} "
              f"{data['timeout_rate_mean']:<7.1f}±{data['timeout_rate_std']:<6.1f} "
              f"{ratio:<10.1f}")

    print("=" * 70)


def save_results(aggregated, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    json_path = os.path.join(output_dir, f"ablation_aggregated_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(convert({"config": config, "results": aggregated, "timestamp": ts}),
                  f, indent=2)
    print(f"\n  JSON: {json_path}")

    csv_path = os.path.join(output_dir, f"ablation_summary_{ts}.csv")
    baseline_ep = aggregated["baseline"]["avg_episode_mean"]
    num_seeds = int(config.get("num_seeds", 1))

    with open(csv_path, "w") as f:
        f.write("method,est_type,window,use_dagger,avg_episode_mean,avg_episode_std,"
                "death_rate_mean,death_rate_std,timeout_rate_mean,timeout_rate_std,"
                "ratio,num_seeds\n")

        f.write(f"Teacher_GT,-,-,-,"
                f"{baseline_ep:.1f},{aggregated['baseline']['avg_episode_std']:.1f},"
                f"{aggregated['baseline']['death_rate_mean']:.2f},"
                f"{aggregated['baseline']['death_rate_std']:.2f},"
                f"{aggregated['baseline']['timeout_rate_mean']:.2f},"
                f"{aggregated['baseline']['timeout_rate_std']:.2f},"
                f"100.0,{num_seeds}\n")

        for name, data in aggregated.items():
            if name == "baseline":
                continue
            ratio = data["avg_episode_mean"] / baseline_ep * 100
            ns = len(data.get("seeds", [1]))
            f.write(f"{name},{data.get('est_type', '?')},{data['window']},"
                    f"{data['use_dagger']},"
                    f"{data['avg_episode_mean']:.1f},{data['avg_episode_std']:.1f},"
                    f"{data['death_rate_mean']:.2f},{data['death_rate_std']:.2f},"
                    f"{data['timeout_rate_mean']:.2f},{data['timeout_rate_std']:.2f},"
                    f"{ratio:.1f},{ns}\n")

    print(f"  CSV: {csv_path}")


# ============================================================================
# MAIN
# ============================================================================

@hydra_task_config(args_cli.task, args_cli.agent_cfg_entry_point)
def main(env_cfg, experiment_cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.events = None  # DR은 AMP 학습 시에만 적용

    print("\n" + "=" * 70)
    print("  SOLO: Ablation Study & Evaluation")
    print("=" * 70)
    if args_cli.run_training:
        print(f"  Mode: TRAIN + EVALUATE (Preset: {PRESET_NAME})")
        print(f"  Seeds: {args_cli.seeds} (from {args_cli.seed_start})")
        print(f"  Collect: {COLLECT_STEPS}, Epochs: {EPOCHS}, DAgger: {args_cli.dagger_rounds} ramp + {args_cli.dagger_extra_rounds} extra")
    else:
        print(f"  Mode: EVALUATE PRE-TRAINED")
        print(f"  Estimator dirs: {args_cli.estimator_dirs}")
    print(f"  Eval: {args_cli.eval_episodes} episodes, max {args_cli.max_episode_steps} steps")
    print(f"  Device: {device}")
    print("=" * 70)

    # ── Environment ──
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    force_skrl_isaaclab_reset(env)
    obs, _ = env.reset()

    # ── Teachers ──
    skrl_runner = Runner(env, experiment_cfg)
    skrl_runner.agent.load(args_cli.teacher_checkpoint)
    skrl_teacher = SkrlAgentWrapper(skrl_runner.agent)

    teacher = TeacherPolicy(OBS_DIM, ACTION_DIM, device=device)
    teacher.load_from_checkpoint(args_cli.teacher_checkpoint, device=device)

    config = {
        "collect_steps": COLLECT_STEPS,
        "epochs": EPOCHS,
        "dagger_rounds": args_cli.dagger_rounds,
        "dagger_extra_rounds": args_cli.dagger_extra_rounds,
        "eval_episodes": args_cli.eval_episodes,
        "max_episode_steps": args_cli.max_episode_steps,
        "hidden_size": args_cli.hidden_size,
        "num_layers": args_cli.num_layers,
        "lr": args_cli.lr,
        "noise_levels": NOISE_LEVELS,
        "num_seeds": args_cli.seeds,
    }

    if args_cli.run_training:
        # ── Train + Evaluate 모든 ablation ──
        model_saver = ModelSaver(args_cli.output_dir, args_cli.teacher_checkpoint)

        all_seed_results = {}
        for seed_idx in range(args_cli.seeds):
            seed = args_cli.seed_start + seed_idx

            print(f"\n{'=' * 70}")
            print(f"  SEED {seed_idx + 1}/{args_cli.seeds}: {seed}")
            print(f"{'=' * 70}")

            seed_results = {}

            # Baseline
            evaluator = Evaluator(50, ENCODER_DIM, PRIV_DIM,
                                  args_cli.max_episode_steps, device)
            baseline = evaluator.evaluate_teacher_gt(
                env, teacher, args_cli.eval_episodes, seed=seed,
            )
            seed_results["baseline"] = baseline
            print(f"  Baseline: episode={baseline['avg_episode']:.1f}, "
                  f"death={baseline['death_rate']:.1f}%")

            # 각 실험
            experiments = get_ablation_experiments(
                skip_history=args_cli.skip_history_ablation,
                skip_mlp=args_cli.skip_mlp,
                skip_tcn=args_cli.skip_tcn,
            )
            for i, (name, est_type, window, use_dagger) in enumerate(experiments):
                print(f"\n  [{i + 1}/{len(experiments)}] {name}")
                result = run_single_experiment(
                    env, teacher, skrl_teacher, device, config,
                    model_saver, name, est_type, window, use_dagger, seed,
                )
                seed_results[name] = result

            all_seed_results[seed] = seed_results

        aggregated = aggregate_results(all_seed_results)

    else:
        # ── Evaluate pre-trained models ──
        if not args_cli.estimator_dirs:
            print("[ERROR] --estimator_dirs required when not using --run_training")
            env.close()
            return

        seeds = list(range(args_cli.seed_start, args_cli.seed_start + args_cli.seeds))

        # Baseline
        evaluator = Evaluator(50, ENCODER_DIM, PRIV_DIM,
                              args_cli.max_episode_steps, device)
        baseline_results = []
        for seed in seeds:
            b = evaluator.evaluate_teacher_gt(env, teacher, args_cli.eval_episodes, seed=seed)
            baseline_results.append(b)

        aggregated = {"baseline": {}}
        for m in ["avg_episode", "std_episode", "death_rate", "timeout_rate"]:
            vals = [b[m] for b in baseline_results]
            aggregated["baseline"][f"{m}_mean"] = np.mean(vals)
            aggregated["baseline"][f"{m}_std"] = np.std(vals)

        pretrained = evaluate_pretrained(
            env, teacher, device, args_cli.estimator_dirs,
            args_cli.eval_episodes, args_cli.max_episode_steps, seeds,
        )

        for name, data in pretrained.items():
            cfg = data["config"]
            aggregated[name] = {
                "window": data["window"],
                "est_type": cfg["type"],
                "use_dagger": True,
                "seeds": seeds,
            }
            for m in ["avg_episode", "std_episode", "death_rate", "timeout_rate"]:
                vals = [r[m] for r in data["seed_results"]]
                aggregated[name][f"{m}_mean"] = np.mean(vals)
                aggregated[name][f"{m}_std"] = np.std(vals)

    # ── Report ──
    print("\n" + "=" * 70)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 70)
    print_summary(aggregated)
    save_results(aggregated, args_cli.output_dir, config)

    print("\n  COMPLETE!")
    print("=" * 70)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
