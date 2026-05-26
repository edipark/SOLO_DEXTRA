# SOLO_ws — Real Robot Deployment

SOLO (State Estimation with Only Leg Observation) 학습 모델을 실제 DEXTRA 하체 로봇에 배포하는 패키지.

## Hardware

- **Robot**: DEXTRA lower-body (12 DOF)
- **Actuators**: Dynamixel AX-18A × 12 (Protocol 1.0)
- **Interface**: Robotis U2D2 USB adapter
- **Computer**: Raspberry Pi 5
- **Sensors**: Joint encoders only (SOLO principle)

## Architecture

```
60Hz Control Loop on RPi 5:
  ┌─────────────────────┐
  │ Dynamixel Read (12) │ → joint_pos (12D)
  │ Finite Diff + EMA   │ → joint_vel (12D)
  │ encoder_obs (24D)   │
  │         ↓           │
  │ LSTM Estimator      │ → priv_est (19D)
  │ (ONNX, 50×24→19)   │
  │         ↓           │
  │ cat(24D, 19D)=43D   │
  │         ↓           │
  │ Teacher Policy      │ → action (12D)
  │ (ONNX, 43→12)      │
  │         ↓           │
  │ Scale + Clip        │ → position targets
  │ Dynamixel Write     │
  └─────────────────────┘
```

## Quick Start

### 1. Export Models (on dev machine with PyTorch)

```bash
cd SOLO_ws
python export_to_onnx.py \
    --teacher_checkpoint ../logs/skrl/dextra_amp_walk/.../best_agent.pt \
    --estimator_checkpoint ../logs/solo_estimator/.../best_estimator.pt \
    --output_dir models/
```

### 2. Setup Raspberry Pi

```bash
# Copy SOLO_ws/ to RPi (via scp, rsync, etc.)
scp -r SOLO_ws/ pi@<rpi-ip>:~/

# On RPi
cd ~/SOLO_ws
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
```

### 3. Test Servos

```bash
python scripts/test_servos.py --config config.yaml           # Ping & read
python scripts/test_servos.py --config config.yaml --sine    # Small oscillation test
python scripts/test_servos.py --config config.yaml --servo-id 1  # Single servo
```

### 4. Deploy

```bash
# Dry-run (inference only, no servo commands)
python deploy.py --config config.yaml --dry-run

# Full deployment
python deploy.py --config config.yaml --log

# With time limit
python deploy.py --config config.yaml --log --duration 30
```

## Calibration

`config.yaml` 의 각 joint `offset_raw` 값은 서보의 0° 위치에 해당하는 raw 값(0–1023).
조립 후 실제 측정이 필요하며, 기본값은 512 (AX-18A center).

## File Structure

```
SOLO_ws/
├── config.yaml              # 로봇/제어 파라미터
├── deploy.py                # 메인 60Hz 제어 루프
├── export_to_onnx.py        # PyTorch → ONNX 변환 (dev machine)
├── requirements.txt
├── hardware/
│   └── dynamixel_interface.py  # AX-18A Protocol 1.0 통신
├── inference/
│   ├── onnx_policy.py          # Teacher policy ONNX wrapper
│   └── onnx_estimator.py       # LSTM estimator + history buffer
├── models/                     # ONNX 모델 파일 (export 후 생성)
├── scripts/
│   ├── setup_pi.sh             # RPi 환경 설정
│   └── test_servos.py          # 서보 테스트
├── utils/
│   ├── timing.py               # 실시간 루프 타이밍
│   └── logger.py               # CSV 로깅
└── logs/                       # 배포 로그 (--log 시 생성)
```

## Safety Features

- **Joint limit clipping**: config.yaml의 upper/lower_rad 범위로 클리핑
- **Action clipping**: [-1, 1] 범위 + action_scale 적용
- **Watchdog**: 루프 >30ms 경고, >100ms 긴급 정지
- **Startup hold**: 시작 시 현재 자세 2초 유지
- **Clean shutdown**: Ctrl+C → home position 복귀 → torque disable
- **Dry-run**: `--dry-run`으로 서보 없이 추론 테스트
