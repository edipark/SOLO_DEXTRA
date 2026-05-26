#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Stiffness Curriculum Trainer for Dextra AMP
============================================
AX-18A 하드웨어 목표: k=11.1 N·m/rad, b=0.2 N·m·s/rad
학습 안정성: k=4.5 → 11.1로 점진적 증가 (b는 0.45 고정)

이 스크립트는 새로운 trainer를 직접 구현하지 않고,
기존 train.py를 subprocess로 호출하며 각 Phase에서
curriculum_stiffness를 Hydra CLI override로 전달합니다.

기본 커리큘럼 스케쥴:
  Phase 0: k=4.5   (80k steps) — warmup, 수렴 기반 확보
  Phase 1: k=6.0   (60k steps)
  Phase 2: k=7.5   (60k steps)
  Phase 3: k=9.0   (60k steps)
  Phase 4: k=11.1  (80k steps) — 하드웨어 목표값

각 Phase는 직전 Phase의 best_agent.pt를 checkpoint로 이어받습니다.

사용법:
  # 기본 실행 (모든 Phase)
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_stiffness_curriculum.py

  # 특정 Phase부터 재개 (Phase 2부터, 직전 checkpoint 경로 지정)
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_stiffness_curriculum.py \\
      --start-phase 2 \\
      --resume-checkpoint logs/skrl/dextra_amp_walk/curriculum_phase1/checkpoints/best_agent.pt

  # Phase 수 / timestep 수 조정
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_stiffness_curriculum.py \\
      --timesteps 60000 60000 60000 60000 80000

  # headless 실행
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_stiffness_curriculum.py \\
      --headless
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default curriculum schedule
# ---------------------------------------------------------------------------
DEFAULT_STIFFNESS = [4.5, 6.0, 7.5, 9.0, 11.1]
DEFAULT_TIMESTEPS = [200000, 80000, 80000, 80000, 100000]
DAMPING = 0.45  # fixed throughout all phases (keeps ζ > 1 even at k=11.1)

TRAIN_SCRIPT = Path(__file__).parent / "train.py"
LOG_ROOT = Path("logs/skrl/dextra_amp_walk")
TASK = "Isaac-Dextra-Amp-Walk-Direct-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stiffness curriculum trainer — wraps train.py across multiple phases."
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        nargs="+",
        default=DEFAULT_STIFFNESS,
        metavar="K",
        help=f"Stiffness values for each phase. Default: {DEFAULT_STIFFNESS}",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=DEFAULT_TIMESTEPS,
        metavar="T",
        help=f"Timesteps for each phase. Default: {DEFAULT_TIMESTEPS}",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=DAMPING,
        help=f"Fixed damping for all phases (default: {DAMPING})",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=0,
        help="Phase index to start from (0-indexed). Use with --resume-checkpoint.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume from at --start-phase.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=TASK,
        help=f"IsaacLab task ID (default: {TASK})",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override number of environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the commands that would be run without executing them.",
    )
    return parser.parse_args()


def phase_log_dir(phase_idx: int, stiffness: float) -> Path:
    """Return the log directory path for a given phase."""
    return LOG_ROOT / f"curriculum_phase{phase_idx}_k{stiffness:.1f}".replace(".", "p")


def find_best_checkpoint(log_dir: Path) -> Path | None:
    """Return best_agent.pt inside log_dir/checkpoints if it exists."""
    candidate = log_dir / "checkpoints" / "best_agent.pt"
    if candidate.exists():
        return candidate
    return None


def build_command(
    phase_idx: int,
    stiffness: float,
    damping: float,
    timesteps: int,
    task: str,
    checkpoint: str | None,
    num_envs: int | None,
    seed: int,
    headless: bool,
    experiment_name: str,
) -> list[str]:
    """Construct the python command for one curriculum phase."""
    isaaclab_sh = Path(__file__).parents[3] / "isaaclab.sh"

    cmd = [
        str(isaaclab_sh), "-p", str(TRAIN_SCRIPT),
        "--task", task,
        "--algorithm", "AMP",
        "--seed", str(seed),
        "--max_iterations", str(timesteps // 16),  # timesteps / rollouts
        # Hydra overrides for actuator curriculum
        f"env.curriculum_stiffness={stiffness}",
        f"env.curriculum_damping={damping}",
        # Tag experiment so logs are easy to identify
        # NOTE: Hydra config root is {"env": ..., "agent": agent_cfg_dict}.
        # agent_cfg_dict itself contains a nested "agent" section, so the
        # correct path is agent.agent.experiment.experiment_name (double "agent").
        f"agent.agent.experiment.experiment_name=curriculum_phase{phase_idx}_k{str(stiffness).replace('.', 'p')}",
    ]

    if checkpoint:
        cmd += ["--checkpoint", str(checkpoint)]
    if num_envs:
        cmd += ["--num_envs", str(num_envs)]
    if headless:
        cmd += ["--headless"]

    return cmd


def run_phase(cmd: list[str], dry_run: bool) -> int:
    """Execute a phase command; return exit code."""
    print("\n" + "=" * 72)
    print("CMD: " + " \\\n     ".join(cmd))
    print("=" * 72 + "\n")

    if dry_run:
        print("[dry-run] Skipping execution.")
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    args = parse_args()

    stiffness_schedule = args.stiffness
    timestep_schedule = args.timesteps

    if len(stiffness_schedule) != len(timestep_schedule):
        print(
            f"[ERROR] --stiffness ({len(stiffness_schedule)} values) and "
            f"--timesteps ({len(timestep_schedule)} values) must have the same length."
        )
        sys.exit(1)

    num_phases = len(stiffness_schedule)
    checkpoint = args.resume_checkpoint  # may be None

    print(f"\n{'='*72}")
    print(f"Stiffness Curriculum: {num_phases} phases")
    print(f"  k schedule: {stiffness_schedule}")
    print(f"  t schedule: {timestep_schedule}")
    print(f"  b (fixed) : {args.damping}")
    print(f"  Start phase: {args.start_phase}")
    if checkpoint:
        print(f"  Resume from: {checkpoint}")
    print(f"{'='*72}\n")

    for phase_idx in range(args.start_phase, num_phases):
        k = stiffness_schedule[phase_idx]
        t = timestep_schedule[phase_idx]
        log_dir = phase_log_dir(phase_idx, k)
        experiment_name = f"curriculum_phase{phase_idx}_k{str(k).replace('.', 'p')}"

        print(f"\n>>> Phase {phase_idx}/{num_phases - 1}  k={k} N·m/rad  b={args.damping} N·m·s/rad  steps={t}")

        cmd = build_command(
            phase_idx=phase_idx,
            stiffness=k,
            damping=args.damping,
            timesteps=t,
            task=args.task,
            checkpoint=checkpoint,
            num_envs=args.num_envs,
            seed=args.seed,
            headless=args.headless,
            experiment_name=experiment_name,
        )

        ret = run_phase(cmd, args.dry_run)

        if ret != 0:
            print(f"\n[ERROR] Phase {phase_idx} exited with code {ret}. Stopping curriculum.")
            sys.exit(ret)

        # Locate the best checkpoint produced by this phase for the next phase.
        # train.py writes logs under logs/skrl/dextra_amp_walk/<timestamp>_amp_torch_<experiment_name>/
        # We search for the most recently modified best_agent.pt under LOG_ROOT.
        best_ckpt = _find_latest_best_checkpoint(LOG_ROOT, experiment_name)
        if best_ckpt:
            checkpoint = str(best_ckpt)
            print(f"  Next phase will resume from: {checkpoint}")
        else:
            print(
                f"  [WARNING] Could not find best_agent.pt for phase {phase_idx} under {LOG_ROOT}. "
                "Next phase will start from scratch."
            )
            checkpoint = None

    print(f"\n{'='*72}")
    print("Curriculum complete.")
    print(f"{'='*72}\n")


def _find_latest_best_checkpoint(log_root: Path, experiment_name_fragment: str) -> Path | None:
    """Search log_root for the most recently modified best_agent.pt whose
    parent directory name contains experiment_name_fragment."""
    candidates = sorted(
        log_root.rglob("best_agent.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if experiment_name_fragment in str(p):
            return p
    # Fallback: return the most recently modified best_agent.pt regardless of name
    if candidates:
        return candidates[0]
    return None


if __name__ == "__main__":
    main()
