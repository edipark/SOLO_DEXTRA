"""해석적 보행 모션 파일 생성기
========================================
사인파 기반 CPG(Central Pattern Generator) 보행 패턴을 이용해
AMP 모션 파일(.npz)을 생성합니다.

주요 파라미터:
  --stride-amp   : Thigh 관절 진폭 [rad]. 클수록 보폭이 커짐.
                   권장: 0.10~0.30 (기존 파일은 ~0.55)
  --walk-speed   : 목표 전진 속도 [m/s].
                   권장: 0.05~0.25 (기존 파일은 ~0.53)
  --step-height  : 스윙 시 발 높이 [m].
                   권장: 0.01~0.04
  --crouch       : 무릎 굽힘 각도 (Calf neutral) [rad].
                   클수록 더 앉은 자세. 권장: 0.4~0.7
  --num-cycles   : 생성할 보행 주기 수. 많을수록 파일이 길어짐.
  --fps          : 출력 fps (sim과 일치시키려면 30).

사용 예:
  # 작은 보폭, 느린 속도 (하드웨어 테스트용)
  python generate_walk_motion.py --stride-amp 0.15 --walk-speed 0.10 --step-height 0.02

  # 중간 보폭
  python generate_walk_motion.py --stride-amp 0.25 --walk-speed 0.18 --step-height 0.03

  # 기존 파일과 비슷한 보폭 (시뮬레이션 전용)
  python generate_walk_motion.py --stride-amp 0.45 --walk-speed 0.40 --step-height 0.04

보행 역학:
  - L/R 다리는 0.5 위상 오프셋 (antipodal gait)
  - Thigh: 사인파 (± stride_amp) + crouch_thigh offset
  - Calf : 항상 양수 (굽힘 유지), 스윙 중 살짝 추가 굽힘
  - AnklePitch: 발 수평 유지 조건 ankle ≈ -(thigh + calf) 기반 + 발목 컴플라이언스
  - HipRoll/HipYaw/AnkleRoll: 소진폭 측방향 안정화
  - body_positions: FK로 계산 (thigh/calf 사용)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# --------------------------------------------------------------------------
# Dextra 링크 파라미터
# --------------------------------------------------------------------------
THIGH_LENGTH = 0.095807   # m
CALF_LENGTH  = 0.093      # m
BASE_HEIGHT  = 0.2865     # m  (URDF init z)

# URDF base → Thigh 관절까지의 z 체인 오프셋
# L_HipYaw origin z = -0.03694
# L_HipRoll origin z = -0.030366
HIP_CHAIN_Z = -0.03694 + (-0.030366)  # = -0.067306 m

# DOF 이름 (기존 .npz 파일과 동일한 순서)
DOF_NAMES = [
    "L_HipYaw_Joint", "R_HipYaw_Joint",
    "L_HipRoll_Joint", "R_HipRoll_Joint",
    "L_Thigh_Joint", "R_Thigh_Joint",
    "L_Calf_Joint", "R_Calf_Joint",
    "L_AnklePitch_Joint", "R_AnklePitch_Joint",
    "L_AnkleRoll_Joint", "R_AnkleRoll_Joint",
]
BODY_NAMES = ["base_link", "L_AnkleRoll_Link_1", "R_AnkleRoll_Link_1"]

# DOF 인덱스
I_L_HIP_YAW   = 0;  I_R_HIP_YAW   = 1
I_L_HIP_ROLL  = 2;  I_R_HIP_ROLL  = 3
I_L_THIGH     = 4;  I_R_THIGH     = 5
I_L_CALF      = 6;  I_R_CALF      = 7
I_L_ANKLE_P   = 8;  I_R_ANKLE_P   = 9
I_L_ANKLE_R   = 10; I_R_ANKLE_R   = 11


# --------------------------------------------------------------------------
# FK: 2D sagittal (x-z)
# --------------------------------------------------------------------------
def leg_fk(thigh: np.ndarray, calf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """관절각 → 발목 위치 (base 기준, sagittal plane).

    Returns:
        foot_x: 전방 변위 (양수=앞)
        foot_z: 수직 변위 (음수=아래)
    """
    knee_x = THIGH_LENGTH * np.sin(thigh)
    knee_z = -THIGH_LENGTH * np.cos(thigh)
    foot_x = knee_x + CALF_LENGTH * np.sin(thigh + calf)
    foot_z = knee_z - CALF_LENGTH * np.cos(thigh + calf)
    return foot_x, foot_z


def leg_height(thigh: np.ndarray, calf: np.ndarray) -> np.ndarray:
    """base_link 기준 발목 높이 (아래가 음수). hip chain z 오프셋 포함."""
    _, foot_z = leg_fk(thigh, calf)
    return HIP_CHAIN_Z + foot_z  # base 기준 실제 발목 높이 (음수)


# --------------------------------------------------------------------------
# 보행 패턴 생성
# --------------------------------------------------------------------------
def generate_gait(
    stride_amp: float,
    walk_speed: float,
    step_height: float,
    crouch: float,
    fps: float,
    num_cycles: int,
    hip_roll_amp: float,
) -> dict:
    """파라미터로 제어 가능한 해석적 보행 모션 딕셔너리 생성."""

    # ---------- 타이밍 ----------
    # 1 stride = 한 발이 앞으로 나가는 거리
    # stride_length ≈ 2 * THIGH_LENGTH * sin(stride_amp)  (간략화)
    stride_length = 2.0 * (
        THIGH_LENGTH * np.sin(stride_amp)
        + CALF_LENGTH * np.sin(stride_amp * 0.5)  # calf 기여
    )
    stride_period = stride_length / max(walk_speed, 1e-4)   # 1 full gait cycle [s]
    stride_period = max(stride_period, 0.4)                  # 최소 0.4s

    # 프레임 수 (마지막 프레임 = 첫 프레임으로 seamless loop)
    frames_per_cycle = max(int(round(stride_period * fps)), 8)
    total_frames = frames_per_cycle * num_cycles + 1

    t = np.linspace(0.0, stride_period * num_cycles, total_frames)
    phase_L = 2.0 * np.pi * t / stride_period          # 왼발 위상
    phase_R = phase_L + np.pi                            # 오른발 (반위상)

    print(f"  stride_length ≈ {stride_length*100:.1f} cm")
    print(f"  stride_period = {stride_period:.3f} s  ({1.0/stride_period:.2f} Hz)")
    print(f"  frames_per_cycle = {frames_per_cycle}, total = {total_frames}")

    # ---------- DOF 초기화 ----------
    dof_pos = np.zeros((total_frames, 12), dtype=np.float32)
    dof_vel = np.zeros((total_frames, 12), dtype=np.float32)

    # --- neutral (stance) 자세 ---
    # 무릎 약간 굽힌 상태: Calf=crouch, Thigh=-crouch/2, Ankle 수평 유지
    thigh_neutral = -crouch * 0.5
    calf_neutral  = crouch
    ankle_neutral = -(thigh_neutral + calf_neutral)  # 발 수평 조건

    # ---------- 각 다리 패턴 ----------
    for side, phase, i_thigh, i_calf, i_ankle_p, i_hip_roll, i_hip_yaw, i_ankle_r in [
        ("L", phase_L, I_L_THIGH, I_L_CALF, I_L_ANKLE_P, I_L_HIP_ROLL, I_L_HIP_YAW, I_L_ANKLE_R),
        ("R", phase_R, I_R_THIGH, I_R_CALF, I_R_ANKLE_P, I_R_HIP_ROLL, I_R_HIP_YAW, I_R_ANKLE_R),
    ]:
        # --- Thigh: 사인파 (앞→뒤) ---
        dof_pos[:, i_thigh] = thigh_neutral + stride_amp * np.sin(phase)

        # --- Calf: 스윙 중 추가 굽힘 (foot clearance) ---
        # sin이 양수(스윙 초기~중간)일 때 추가 굽힘
        swing_extra = step_height * 8.0   # step_height → calf 추가 굽힘 환산
        swing_extra = np.clip(swing_extra, 0.0, 0.4)
        calf_swing = calf_neutral + swing_extra * np.clip(np.sin(phase), 0.0, 1.0)
        dof_pos[:, i_calf] = calf_swing

        # --- AnklePitch: 발 수평 유지 + 발끝 띄우기 ---
        # 기본: 발 수평 조건 ankle = -(thigh + calf)
        # 스윙 중 발끝을 약간 올림 (dorsiflexion)
        toe_up = 0.05 * np.clip(np.sin(phase), 0.0, 1.0)
        dof_pos[:, i_ankle_p] = -(dof_pos[:, i_thigh] + dof_pos[:, i_calf]) + toe_up

        # --- HipRoll: 측방향 무게 이동 (반위상) ---
        # 지지발 쪽으로 약간 기울임
        dof_pos[:, i_hip_roll] = -hip_roll_amp * np.sin(phase)

        # --- HipYaw: 걸음에 따른 소진폭 회전 ---
        dof_pos[:, i_hip_yaw] = (stride_amp * 0.05) * np.sin(phase)

        # --- AnkleRoll: HipRoll 보상 ---
        dof_pos[:, i_ankle_r] = hip_roll_amp * 0.3 * np.sin(phase)

    # ---------- 속도: 유한 차분 ----------
    dt = 1.0 / fps
    dof_vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) / (2.0 * dt)
    dof_vel[0]    = (dof_pos[1] - dof_pos[0]) / dt
    dof_vel[-1]   = (dof_pos[-1] - dof_pos[-2]) / dt

    # ---------- body_positions (FK) ----------
    body_pos = np.zeros((total_frames, 3, 3), dtype=np.float32)  # (N, 3bodies, xyz)

    thigh_L = dof_pos[:, I_L_THIGH]
    calf_L  = dof_pos[:, I_L_CALF]
    thigh_R = dof_pos[:, I_R_THIGH]
    calf_R  = dof_pos[:, I_R_CALF]

    # base_link: 전진 + 높이
    leg_h_L = leg_height(thigh_L, calf_L)  # base 기준 발목 z (음수=아래)
    leg_h_R = leg_height(thigh_R, calf_R)
    # base_z = 지면 기준 base 높이
    # 지지발(더 아래로 뻗은 발)이 지면(z=0)에 닿는다고 가정 → max(-foot_z) 사용
    # 평균 사용 시 지지발이 바닥 아래로 내려가는 버그 발생
    base_z = np.maximum(-leg_h_L, -leg_h_R)

    # base x: 속도 적분
    base_x = walk_speed * t
    base_x = base_x - base_x[0]  # 첫 프레임 기준 상대

    body_pos[:, 0, 0] = base_x.astype(np.float32)
    body_pos[:, 0, 2] = base_z.astype(np.float32)
    # y는 0 (측면 이동 없음)

    # L ankle 위치 (base 기준)
    foot_x_L, foot_z_L = leg_fk(thigh_L, calf_L)
    body_pos[:, 1, 0] = (base_x + foot_x_L).astype(np.float32)
    body_pos[:, 1, 1] = 0.04435  # hip Y 오프셋 (URDF에서)
    body_pos[:, 1, 2] = (base_z + foot_z_L).astype(np.float32)

    # R ankle 위치 (base 기준)
    foot_x_R, foot_z_R = leg_fk(thigh_R, calf_R)
    body_pos[:, 2, 0] = (base_x + foot_x_R).astype(np.float32)
    body_pos[:, 2, 1] = -0.04435
    body_pos[:, 2, 2] = (base_z + foot_z_R).astype(np.float32)

    # ---------- body_rotations: slight pitch for body forward motion ----------
    # body_rotations 형식: (N, 3bodies, 4) wxyz quaternions
    # 대부분 identity이지만, base_link는 약간의 pitch (앞으로 기울임)
    # pitch ~= 0.02 rad (1도) 정도로 자연스러운 보행 모습
    from scipy.spatial.transform import Rotation
    
    body_rot = np.zeros((total_frames, 3, 4), dtype=np.float32)
    
    # base_link: 약간의 forward pitch
    pitch = 0.02  # 라디안
    r_base = Rotation.from_euler('y', pitch)  # y축 회전 (pitch)
    q_base_xyzw = r_base.as_quat()  # scipy 기본: xyzw 형식
    q_base_wxyz = np.array([q_base_xyzw[3], q_base_xyzw[0], q_base_xyzw[1], q_base_xyzw[2]])
    body_rot[:, 0, :] = q_base_wxyz.astype(np.float32)
    
    # L/R ankles: identity (feet는 수평)
    body_rot[:, 1, 0] = 1.0  # L ankle w=1
    body_rot[:, 2, 0] = 1.0  # R ankle w=1

    # ---------- body 선속도 (유한 차분) ----------
    body_lin_vel = np.zeros((total_frames, 3, 3), dtype=np.float32)
    body_lin_vel[1:-1] = (body_pos[2:] - body_pos[:-2]) / (2.0 * dt)
    body_lin_vel[0]    = (body_pos[1] - body_pos[0]) / dt
    body_lin_vel[-1]   = (body_pos[-1] - body_pos[-2]) / dt

    # ---------- body 각속도 (0으로 근사) ----------
    body_ang_vel = np.zeros((total_frames, 3, 3), dtype=np.float32)

    # ---------- seamless loop: 마지막 프레임 = 첫 프레임 ----------
    # (num_cycles 정수 주기 → 이미 seamless지만 명시적으로 적용)
    dof_pos[-1]     = dof_pos[0]
    dof_vel[-1]     = dof_vel[0]
    body_rot[-1]    = body_rot[0]
    body_ang_vel[-1]= body_ang_vel[0]
    # L/R ankle 위치도 첫 프레임 기준으로 리셋 (주기 완성)
    # base_link x는 선형 진행을 나타내므로 마지막은 첫 프레임보다 전진해있음
    # 이것은 의도적: 모션이 반복될 때마다 로봇이 계속 앞으로 나감
    body_pos[-1, 1:, :] = body_pos[0, 1:, :]
    body_lin_vel[-1]     = body_lin_vel[0]

    return {
        "fps":                    np.float64(fps),
        "dof_names":              np.array(DOF_NAMES),
        "body_names":             np.array(BODY_NAMES),
        "dof_positions":          dof_pos,
        "dof_velocities":         dof_vel,
        "body_positions":         body_pos,
        "body_rotations":         body_rot,
        "body_linear_velocities": body_lin_vel,
        "body_angular_velocities":body_ang_vel,
    }


# --------------------------------------------------------------------------
# 진단 출력
# --------------------------------------------------------------------------
def print_stats(data: dict) -> None:
    pos = data["dof_positions"]
    names = data["dof_names"].tolist()
    print("\n=== 생성된 모션 관절 범위 ===")
    for i, n in enumerate(names):
        mn, mx = pos[:, i].min(), pos[:, i].max()
        amp = mx - mn
        flag = " !!!" if amp > 1.0 else ""
        print(f"  {n:25s}: [{mn:+.4f}, {mx:+.4f}]  amp={amp:.4f}{flag}")

    bpos = data["body_positions"]
    bvel = data["body_linear_velocities"]
    fps = float(data["fps"])
    N = pos.shape[0]
    duration = (N - 1) / fps
    x_total = bpos[-1, 0, 0] - bpos[0, 0, 0]
    mean_vx = bvel[:, 0, 0].mean()
    print(f"\n=== 모션 요약 ===")
    print(f"  frames={N}, fps={fps:.0f}, duration={duration:.3f}s")
    print(f"  전진 거리={x_total:.4f}m, 평균 vx={mean_vx:.4f}m/s")
    base_z = bpos[:, 0, 2]
    print(f"  base_link z: [{base_z.min():.4f}, {base_z.max():.4f}]")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="해석적 보행 모션 파일 생성기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stride-amp", type=float, default=0.20,
                        help="Thigh 관절 진폭 [rad]. 클수록 보폭 큼 (권장: 0.10~0.30)")
    parser.add_argument("--walk-speed", type=float, default=0.12,
                        help="전진 속도 [m/s] (권장: 0.05~0.25)")
    parser.add_argument("--step-height", type=float, default=0.025,
                        help="스윙 발 높이 [m] (권장: 0.01~0.04)")
    parser.add_argument("--crouch", type=float, default=0.55,
                        help="중립 Calf 굽힘 [rad]. 클수록 더 앉은 자세 (권장: 0.4~0.7)")
    parser.add_argument("--hip-roll-amp", type=float, default=0.06,
                        help="HipRoll 진폭 [rad]. 측방향 무게이동 (권장: 0.02~0.10)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="출력 fps")
    parser.add_argument("--num-cycles", type=int, default=4,
                        help="반복 주기 수")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 파일명 (None이면 자동 생성)")
    args = parser.parse_args()

    print("=== 보행 모션 생성 ===")
    print(f"  stride_amp   = {args.stride_amp:.3f} rad")
    print(f"  walk_speed   = {args.walk_speed:.3f} m/s")
    print(f"  step_height  = {args.step_height:.3f} m")
    print(f"  crouch(Calf) = {args.crouch:.3f} rad")
    print(f"  hip_roll_amp = {args.hip_roll_amp:.3f} rad")
    print(f"  fps          = {args.fps:.0f}")
    print(f"  num_cycles   = {args.num_cycles}")

    data = generate_gait(
        stride_amp  = args.stride_amp,
        walk_speed  = args.walk_speed,
        step_height = args.step_height,
        crouch      = args.crouch,
        fps         = args.fps,
        num_cycles  = args.num_cycles,
        hip_roll_amp= args.hip_roll_amp,
    )

    print_stats(data)

    # 출력 파일명 자동 생성
    if args.output is None:
        stride_str = f"{args.stride_amp:.2f}".replace(".", "p")
        speed_str  = f"{args.walk_speed:.2f}".replace(".", "p")
        fname = f"dextra_walk_analytic_stride{stride_str}_speed{speed_str}.npz"
    else:
        fname = args.output

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, fname)

    np.savez_compressed(out_path, **data)
    print(f"\n[생성 완료] {out_path}")
    print(f"  env_cfg.motion_file = \"{out_path}\"  으로 사용하세요.")


if __name__ == "__main__":
    main()
