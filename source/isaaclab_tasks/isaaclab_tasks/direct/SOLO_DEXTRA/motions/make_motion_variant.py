"""
모션 파일 변형 스크립트 (통합)
==============================
기존 npz 모션 파일에서 시간 리샘플링(fps/속도), 보폭(stride), 속도(vel)를
조합해 새로운 모션 파일을 생성한다.

기존 두 스크립트의 기능을 통합:
  - create_motion_variant.py: 시간축 리샘플링 (fps 변환 + 재생 속도)
  - make_motion_variant.py:   보폭/속도 수학적 스케일링

사용 예:
  # 1) fps 변환 + 3배 느리게
  python make_motion_variant.py \\
      --input dextra_walk_flat_pitch_fk.npz \\
      --target-fps 30 --speed-scale 3.0

  # 2) 보폭 60%로 줄이기
  python make_motion_variant.py \\
      --input dextra_walk_flat_pitch_fk_30hz_3p0x_slower.npz \\
      --stride-scale 0.6

  # 3) 속도 필드 40%로 감쇄
  python make_motion_variant.py \\
      --input dextra_walk_flat_pitch_fk_30hz_3p0x_slower_stride0p60.npz \\
      --vel-scale 0.4

  # 4) 전체 파이프라인 한 번에
  python make_motion_variant.py \
      --input dextra_walk_flat_pitch_fk.npz \
      --target-fps 30 --speed-scale 3.0 --stride-scale 0.6 --vel-scale 0.4

  # 5) 루프 경계 불연속 제거 (비주기 모션 → 주기화)
  python make_motion_variant.py \\
      --input dextra_walk_flat_pitch_fk_30hz_2x_slower_stride0p6_vel0p8.npz \\
      --make-periodic

  # 6) 모든 변환 + 주기화
  python source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/motions/make_motion_variant.py \
      --input dextra_walk_flat_pitch_fk.npz \
      --target-fps 30 --speed-scale 2.0 --stride-scale 0.6 --vel-scale 0.8 --make-periodic

  # 7) 대칭화 (좌우 관절 진폭/평균 균등화)
  python source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/motions/make_motion_variant.py \
      --input dextra_walk_flat_pitch_fk.npz \
      --target-fps 30 --speed-scale 2.0 --stride-scale 0.6 --vel-scale 0.8 --make-periodic --symmetrize right

조정 방식:
  --target-fps / --speed-scale:
      시간축을 리샘플링해 fps를 바꾸고 재생 속도를 조절한다.
      위치/회전은 선형 보간 + SLERP, 속도는 원본 보간 후 1/speed_scale 스케일.
      speed_scale > 1 이면 느려짐(duration 증가), < 1 이면 빨라짐.
  --stride-scale:
      Thigh/Calf/AnklePitch 관절의 진폭을 mean 중심으로 스케일.
      base 진행량과 발 위치의 전방(x) 성분도 동일 비율 조정.
  --vel-scale:
      body_linear_velocities x 성분과 body_positions x 진행량을 독립적으로 스케일.
      dof_velocities는 변경하지 않음(관절 속도 그대로).
  --make-periodic:
      루프 경계 불연속을 제거해 모션을 주기화한다.
      dof_positions 기준으로 첫 프레임과 가장 가까운 프레임을 탐색해 해당 지점까지 자르고,
      마지막 프레임을 첫 프레임과 동일하게 교체해 seamless loop를 만든다.
      모든 다른 변환이 완료된 후 마지막 단계로 적용된다.
  --symmetrize [avg|left|right]:
      좌우 관절 쌍의 진폭(amplitude)과 평균(mean)을 대칭화한다.
      dof_positions/dof_velocities의 모든 L/R 쌍에 적용.
        avg   (기본값): L/R 진폭과 평균의 산술 평균으로 양쪽을 통일.
        left:  L 기준으로 R을 맞춤.
        right: R 기준으로 L을 맞춤.
      적용 순서: resample → stride → vel → symmetrize → make-periodic.
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# 보폭에 주로 기여하는 관절 인덱스 (Thigh, Calf, AnklePitch 좌우)
# HipYaw/HipRoll/AnkleRoll는 측방향 안정성 → 스케일 제외
STRIDE_JOINT_INDICES = [4, 5, 6, 7, 8, 9]

# 좌우 관절 쌍 인덱스: (L, R)
# [L_HipYaw, R_HipYaw, L_HipRoll, R_HipRoll, L_Thigh, R_Thigh,
#  L_Calf, R_Calf, L_AnklePitch, R_AnklePitch, L_AnkleRoll, R_AnkleRoll]
JOINT_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]


# ---------------------------------------------------------------------------
# 시간축 리샘플링 (create_motion_variant.py 기능)
# ---------------------------------------------------------------------------

def _slerp_sequence(rotations_wxyz: np.ndarray, t_orig: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """SLERP로 wxyz 쿼터니언 시퀀스를 보간한다."""
    rots_xyzw = rotations_wxyz[:, [1, 2, 3, 0]]
    slerp_fn = Slerp(t_orig, Rotation.from_quat(rots_xyzw))
    result_xyzw = slerp_fn(t_query).as_quat()
    return result_xyzw[:, [3, 0, 1, 2]]


def resample_motion(data: dict, target_fps: float, speed_scale: float) -> dict:
    """시간축을 리샘플링해 fps와 재생 속도를 변경한다.

    위치/회전: 보간(선형/SLERP).
    속도: 원본 속도 보간 후 1/speed_scale 스케일
    (위치 유한차분을 쓰면 contact staircase 때문에 노이즈가 심함).
    """
    orig_fps: float = float(data["fps"])
    orig_frames: int = int(data["dof_positions"].shape[0])
    orig_duration: float = (orig_frames - 1) / orig_fps

    new_duration: float = orig_duration * speed_scale
    new_dt: float = 1.0 / target_fps
    new_frames: int = round(new_duration / new_dt) + 1

    print(f"  원본: fps={orig_fps:.1f}, frames={orig_frames}, duration={orig_duration:.3f}s")
    print(f"  변환: fps={target_fps:.1f}, speed_scale={speed_scale:.2f}x")
    print(f"        frames={new_frames}, duration={new_duration:.3f}s")

    t_orig = np.linspace(0.0, orig_duration, orig_frames)
    t_new = np.linspace(0.0, new_duration, new_frames)
    t_sample = np.clip(t_new / speed_scale, 0.0, orig_duration)

    def interp1d_array(arr2d: np.ndarray) -> np.ndarray:
        """(N, D) 배열을 각 차원별로 np.interp."""
        return np.stack(
            [np.interp(t_sample, t_orig, arr2d[:, j].astype(np.float64))
             for j in range(arr2d.shape[1])],
            axis=1,
        ).astype(np.float32)

    def interp_body(arr3d: np.ndarray) -> np.ndarray:
        """(N, B, 3) 배열을 각 바디/축별로 np.interp."""
        B = arr3d.shape[1]
        return np.stack(
            [np.stack(
                [np.interp(t_sample, t_orig, arr3d[:, b, xyz].astype(np.float64))
                 for xyz in range(3)],
                axis=-1,
            ) for b in range(B)],
            axis=1,
        ).astype(np.float32)

    out = dict(data)
    out["fps"] = np.float64(target_fps)
    out["dof_positions"] = interp1d_array(data["dof_positions"])
    out["dof_velocities"] = interp1d_array(data["dof_velocities"]) / np.float32(speed_scale)
    out["body_positions"] = interp_body(data["body_positions"])
    out["body_linear_velocities"] = interp_body(data["body_linear_velocities"]) / np.float32(speed_scale)
    out["body_angular_velocities"] = interp_body(data["body_angular_velocities"]) / np.float32(speed_scale)

    # 회전: 바디별 SLERP
    B = data["body_rotations"].shape[1]
    out["body_rotations"] = np.stack(
        [_slerp_sequence(data["body_rotations"][:, b, :].astype(np.float64), t_orig, t_sample)
         for b in range(B)],
        axis=1,
    ).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# 보폭 스케일링 (make_motion_variant.py 기능)
# ---------------------------------------------------------------------------

def scale_stride(data: dict, stride_scale: float) -> dict:
    """Thigh/Calf/AnklePitch 관절 진폭과 발 전방 위치를 stride_scale로 스케일한다."""
    out = {k: v.copy() for k, v in data.items()}

    dof_pos = out["dof_positions"].copy()
    dof_vel = out["dof_velocities"].copy()
    for idx in STRIDE_JOINT_INDICES:
        mean = dof_pos[:, idx].mean()
        dof_pos[:, idx] = (dof_pos[:, idx] - mean) * stride_scale + mean
        dof_vel[:, idx] *= stride_scale
    out["dof_positions"] = dof_pos
    out["dof_velocities"] = dof_vel

    body_pos = out["body_positions"].copy()
    base_x = body_pos[:, 0, 0].copy()
    start_x = base_x[0]
    body_pos[:, 0, 0] = (base_x - start_x) * stride_scale + start_x
    new_base_x = body_pos[:, 0, 0]
    for b in [1, 2]:
        rel_x = body_pos[:, b, 0] - base_x
        body_pos[:, b, 0] = rel_x * stride_scale + new_base_x
    out["body_positions"] = body_pos

    body_vel = out["body_linear_velocities"].copy()
    body_vel[:, :, 0] *= stride_scale
    out["body_linear_velocities"] = body_vel

    return out


# ---------------------------------------------------------------------------
# 속도 스케일링 (make_motion_variant.py 기능)
# ---------------------------------------------------------------------------

def scale_velocity(data: dict, vel_scale: float) -> dict:
    """body_linear_velocities x 성분과 body_positions x 진행량을 vel_scale로 스케일한다.

    dof_velocities(관절 속도)는 변경하지 않는다.
    """
    out = {k: v.copy() for k, v in data.items()}

    body_vel = out["body_linear_velocities"].copy()
    body_vel[:, :, 0] *= vel_scale
    out["body_linear_velocities"] = body_vel

    body_pos = out["body_positions"].copy()
    base_x = body_pos[:, 0, 0].copy()
    start_x = base_x[0]
    new_base_x = (base_x - start_x) * vel_scale + start_x
    delta = new_base_x - base_x
    body_pos[:, :, 0] += delta[:, np.newaxis]
    out["body_positions"] = body_pos

    return out


# ---------------------------------------------------------------------------
# 대칭화 (symmetrize)
# ---------------------------------------------------------------------------

def symmetrize_motion(data: dict, mode: str = "avg") -> dict:
    """좌우 관절 쌍의 진폭(amplitude)과 평균(mean)을 대칭화한다.

    mode:
      'avg'   : L/R 진폭과 평균의 산술 평균으로 양쪽을 통일 (기본값)
      'left'  : L 기준으로 R을 맞춤
      'right' : R 기준으로 L을 맞춤

    적용 대상:
      - dof_positions / dof_velocities: 모든 12개 관절 L/R 쌍
      - body_positions x 성분: L/R 발 위치
      - body_linear_velocities x 성분: L/R 발 속도 (body index 1, 2)
    """
    if mode not in ("avg", "left", "right"):
        raise ValueError(f"--symmetrize mode must be one of avg/left/right, got: {mode}")

    out = {k: v.copy() for k, v in data.items()}

    dof_pos = out["dof_positions"].copy()   # (N, 12)
    dof_vel = out["dof_velocities"].copy()  # (N, 12)

    print(f"  대칭화 모드: {mode}")
    print(f"  {'관절 쌍':40s}  {'L amp':>8}  {'R amp':>8}  {'→ target amp':>12}  {'L mean':>8}  {'R mean':>8}  {'→ target mean':>13}")

    for l_idx, r_idx in JOINT_PAIRS:
        l_pos = dof_pos[:, l_idx]
        r_pos = dof_pos[:, r_idx]

        l_mean = float(l_pos.mean())
        r_mean = float(r_pos.mean())
        l_amp  = float(l_pos.max() - l_pos.min())
        r_amp  = float(r_pos.max() - r_pos.min())

        if mode == "avg":
            tgt_amp  = (l_amp + r_amp) / 2.0
            tgt_mean = (l_mean + r_mean) / 2.0
        elif mode == "left":
            tgt_amp  = l_amp
            tgt_mean = l_mean
        else:  # right
            tgt_amp  = r_amp
            tgt_mean = r_mean

        pair_name = f"[{l_idx},{r_idx}] pair"
        print(f"  {pair_name:40s}  {l_amp:8.4f}  {r_amp:8.4f}  {tgt_amp:12.4f}  {l_mean:8.4f}  {r_mean:8.4f}  {tgt_mean:13.4f}")

        # 진폭 스케일 + 평균 이동
        for (idx, cur_amp, cur_mean) in ((l_idx, l_amp, l_mean), (r_idx, r_amp, r_mean)):
            if cur_amp > 1e-6:
                scale = tgt_amp / cur_amp
                dof_pos[:, idx] = (dof_pos[:, idx] - cur_mean) * scale + tgt_mean
                dof_vel[:, idx] *= scale
            else:
                # 진폭이 거의 없으면 평균만 이동
                dof_pos[:, idx] += (tgt_mean - cur_mean)

    out["dof_positions"] = dof_pos
    out["dof_velocities"] = dof_vel

    # body_positions L/R 발 x 성분 대칭화 (body index 1 = L, 2 = R)
    bp = out["body_positions"].copy()  # (N, B, 3)
    bv = out["body_linear_velocities"].copy()  # (N, B, 3)
    base_x = bp[:, 0, 0]

    for foot_l, foot_r in ((1, 2),):
        l_rel = bp[:, foot_l, 0] - base_x
        r_rel = bp[:, foot_r, 0] - base_x

        l_amp_b  = float(l_rel.max() - l_rel.min())
        r_amp_b  = float(r_rel.max() - r_rel.min())
        l_mean_b = float(l_rel.mean())
        r_mean_b = float(r_rel.mean())

        if mode == "avg":
            tgt_amp_b  = (l_amp_b + r_amp_b) / 2.0
            tgt_mean_b = (l_mean_b + r_mean_b) / 2.0
        elif mode == "left":
            tgt_amp_b, tgt_mean_b = l_amp_b, l_mean_b
        else:
            tgt_amp_b, tgt_mean_b = r_amp_b, r_mean_b

        for (fidx, cur_amp_b, cur_mean_b) in ((foot_l, l_amp_b, l_mean_b), (foot_r, r_amp_b, r_mean_b)):
            if cur_amp_b > 1e-6:
                scale_b = tgt_amp_b / cur_amp_b
                bp[:, fidx, 0] = (bp[:, fidx, 0] - (cur_mean_b + base_x)) * scale_b + (tgt_mean_b + base_x)
                bv[:, fidx, 0] *= scale_b

        print(f"  발 X 진폭: L={l_amp_b:.4f}m  R={r_amp_b:.4f}m  → target={tgt_amp_b:.4f}m")

    out["body_positions"] = bp
    out["body_linear_velocities"] = bv
    return out


# ---------------------------------------------------------------------------
# 주기화 (make-periodic)
# ---------------------------------------------------------------------------

def make_periodic(data: dict) -> dict:
    """루프 경계 불연속을 제거해 모션을 주기화한다.

    알고리즘:
      1. dof_positions의 첫 프레임을 기준으로, 전체 프레임 중 1/3 지점부터 끝까지
         각 프레임과의 L2 거리를 계산한다.
      2. 거리가 가장 작은 프레임을 루프 포인트로 선택한다.
      3. 선택된 프레임까지 잘라내고, 마지막 프레임을 첫 프레임으로 교체해
         seamless loop를 만든다. body_positions의 X 누적 진행량은 보존한다.
    """
    dof = data["dof_positions"]       # (N, D)
    n = dof.shape[0]
    start = n // 3                     # 첫 1/3은 제외 (너무 가까운 위치 방지)

    # 첫 프레임과의 거리 (dof_positions만 사용; 가장 의미있는 state)
    ref = dof[0]                       # (D,)
    dists = np.linalg.norm(dof[start:] - ref, axis=1)  # (N - start,)
    loop_idx = int(np.argmin(dists)) + start

    print(f"  루프 포인트 탐색 범위: frame {start} ~ {n-1}")
    print(f"  최적 루프 포인트: frame {loop_idx}  (첫 프레임과의 dof L2 거리={dists[loop_idx-start]:.4f})")

    # frame 0 ~ loop_idx 까지 자르기 (loop_idx 포함)
    def trim(arr: np.ndarray) -> np.ndarray:
        return arr[:loop_idx + 1].copy()

    out = {k: trim(v) if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n) else v.copy()
           for k, v in data.items()}

    # 마지막 프레임 → 첫 프레임으로 교체 (seamless)
    # body_positions의 X 진행량은 loop_idx 시점의 절대 위치를 기준으로 유지
    out["dof_positions"][-1]      = data["dof_positions"][0].copy()
    out["dof_velocities"][-1]     = data["dof_velocities"][0].copy()
    out["body_rotations"][-1]     = data["body_rotations"][0].copy()
    out["body_angular_velocities"][-1] = data["body_angular_velocities"][0].copy()
    # linear_velocities: 마지막 → 첫 프레임 속도로 교체
    out["body_linear_velocities"][-1] = data["body_linear_velocities"][0].copy()
    # body_positions: X 진행량을 유지하면서 Z/Y는 첫 프레임과 맞춤
    for b in range(out["body_positions"].shape[1]):
        out["body_positions"][-1, b, 1] = data["body_positions"][0, b, 1]  # Y
        out["body_positions"][-1, b, 2] = data["body_positions"][0, b, 2]  # Z
        # X는 이전 프레임에서 부드럽게 이어지도록 그대로 둠

    new_n = out["dof_positions"].shape[0]
    fps = float(out["fps"])
    print(f"  트리밍 후: {n} → {new_n} frames ({(new_n-1)/fps:.3f}s)")

    # 주기성 검증
    diff = np.abs(out["dof_positions"][-1] - out["dof_positions"][0]).max()
    print(f"  첫/끝 dof max 차이 (교체 후): {diff:.6f} rad  (이상적=0)")

    return out


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def _fmt(val: float) -> str:
    """숫자를 파일명용 문자열로 변환 (1.0→'1p0', 0.4→'0p4')."""
    return f"{val:.2f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def main():
    parser = argparse.ArgumentParser(
        description="모션 파일 변형 도구 (시간 리샘플링 + 보폭/속도 스케일링 통합)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="입력 npz 파일명 (스크립트와 같은 디렉토리 기준 또는 절대 경로)")
    parser.add_argument("--output", default=None,
                        help="출력 npz 파일명 (미지정시 자동 생성)")

    # 시간 리샘플링
    parser.add_argument("--target-fps", type=float, default=None,
                        help="목표 fps (미지정시 원본 fps 유지)")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help=">1 이면 느려짐(duration 증가), <1 이면 빨라짐 (기본값 1.0)")

    # 보폭/속도 스케일
    parser.add_argument("--stride-scale", type=float, default=1.0,
                        help="보폭 스케일 (0.6 = 60%%, 기본값 1.0)")
    parser.add_argument("--vel-scale", type=float, default=1.0,
                        help="베이스 전방 속도 스케일 (0.4 = 40%%, 기본값 1.0)")
    parser.add_argument("--make-periodic", action="store_true", default=False,
                        help="루프 경계 불연속 제거: 최적 루프 포인트에서 자르고 끝 프레임을 첫 프레임으로 교체")
    parser.add_argument("--symmetrize", nargs="?", const="avg", default=None,
                        metavar="MODE",
                        help="좌우 관절 쌍 진폭/평균 대칭화. MODE=avg(기본)|left|right")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(script_dir, args.input)

    if not os.path.exists(input_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {input_path}")
        return

    # 로드
    raw = np.load(input_path, allow_pickle=False)
    data = {k: raw[k].copy() for k in raw.files}

    src_fps = float(data["fps"])
    src_frames = data["dof_positions"].shape[0]
    src_dur = (src_frames - 1) / src_fps
    src_vx = data["body_linear_velocities"][:, 0, 0]

    print(f"\n{'='*60}")
    print(f"[입력] {os.path.basename(input_path)}")
    print(f"  fps={src_fps:.1f}, frames={src_frames}, duration={src_dur:.3f}s")
    print(f"  base vx: min={src_vx.min():.4f}  max={src_vx.max():.4f}  mean={src_vx.mean():.4f} m/s")
    print(f"  L_foot x 상대범위: "
          f"{(data['body_positions'][:,1,0]-data['body_positions'][:,0,0]).min():.4f} ~ "
          f"{(data['body_positions'][:,1,0]-data['body_positions'][:,0,0]).max():.4f} m")
    print(f"{'='*60}")

    suffix_parts = []

    # 1. 시간 리샘플링
    target_fps = args.target_fps if args.target_fps is not None else src_fps
    need_resample = (target_fps != src_fps) or (args.speed_scale != 1.0)
    if need_resample:
        print(f"\n[1/5] 시간 리샘플링")
        data = resample_motion(data, target_fps, args.speed_scale)
        if target_fps != src_fps:
            suffix_parts.append(f"{int(target_fps)}hz")
        if args.speed_scale != 1.0:
            suffix_parts.append(f"{_fmt(args.speed_scale)}x_slower"
                                 if args.speed_scale > 1.0
                                 else f"{_fmt(args.speed_scale)}x_faster")
    else:
        print(f"\n[1/5] 시간 리샘플링: 건너뜀 (target-fps={target_fps:.1f}, speed-scale=1.0)")

    # 2. 보폭 스케일
    if args.stride_scale != 1.0:
        print(f"\n[2/5] 보폭 스케일: {args.stride_scale}")
        old_foot = data["body_positions"][:, 1, 0] - data["body_positions"][:, 0, 0]
        data = scale_stride(data, args.stride_scale)
        new_foot = data["body_positions"][:, 1, 0] - data["body_positions"][:, 0, 0]
        print(f"  L_foot x 상대범위: {old_foot.min():.4f}~{old_foot.max():.4f}"
              f" → {new_foot.min():.4f}~{new_foot.max():.4f} m")
        suffix_parts.append(f"stride{_fmt(args.stride_scale)}")
    else:
        print(f"\n[2/5] 보폭 스케일: 건너뜀 (stride-scale=1.0)")

    # 3. 속도 스케일
    if args.vel_scale != 1.0:
        print(f"\n[3/5] 속도 스케일: {args.vel_scale}")
        old_vx = data["body_linear_velocities"][:, 0, 0]
        print(f"  base vx before: min={old_vx.min():.4f}  max={old_vx.max():.4f}  mean={old_vx.mean():.4f}")
        data = scale_velocity(data, args.vel_scale)
        new_vx = data["body_linear_velocities"][:, 0, 0]
        print(f"  base vx after:  min={new_vx.min():.4f}  max={new_vx.max():.4f}  mean={new_vx.mean():.4f}")
        suffix_parts.append(f"vel{_fmt(args.vel_scale)}")
    else:
        print(f"\n[3/5] 속도 스케일: 건너뜀 (vel-scale=1.0)")

    # 4. 대칭화
    if args.symmetrize is not None:
        sym_mode = args.symmetrize
        print(f"\n[4/5] 대칭화 (--symmetrize {sym_mode})")
        data = symmetrize_motion(data, mode=sym_mode)
        suffix_parts.append(f"sym{sym_mode}" if sym_mode != "avg" else "sym")
    else:
        print(f"\n[4/5] 대칭화: 건너뜀 (--symmetrize 미지정)")

    # 5. 주기화 (항상 마지막 단계)
    if args.make_periodic:
        print(f"\n[5/5] 주기화 (--make-periodic)")
        data = make_periodic(data)
        suffix_parts.append("periodic")
    else:
        print(f"\n[5/5] 주기화: 건너뜀 (--make-periodic 미지정)")

    # 출력 파일명
    if args.output is not None:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(script_dir, args.output)
    else:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else "_copy"
        output_path = os.path.join(script_dir, stem + suffix + ".npz")

    np.savez(output_path, **data)

    final_vx = data["body_linear_velocities"][:, 0, 0]
    final_frames = data["dof_positions"].shape[0]
    final_fps = float(data["fps"])
    final_dur = (final_frames - 1) / final_fps
    print(f"\n{'='*60}")
    print(f"[출력] {os.path.basename(output_path)}")
    print(f"  fps={final_fps:.1f}, frames={final_frames}, duration={final_dur:.3f}s")
    print(f"  base vx: min={final_vx.min():.4f}  max={final_vx.max():.4f}  mean={final_vx.mean():.4f} m/s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
