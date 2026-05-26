# Motor System Identification (SysID) for AX-18A → ImplicitActuatorCfg

## 목적

실제 Dynamixel AX-18A 서보의 step response를 측정하여 IsaacLab
`ImplicitActuatorCfg`의 `stiffness`(k)와 `damping`(d)을 실제 하드웨어에
맞게 보정한다.

```
real robot → step response 측정
          → 2nd-order PD 모델 fit (ωn, ζ)
          → k = J·ωn²,  d = 2ζωn·J
          → dextra_amp_env_cfg.py 업데이트
          → teacher 재학습
```

---

## 배경: ImplicitActuator 모델

IsaacLab의 `ImplicitActuatorCfg`는 PhysX 내부에서 연속시간 PD로 동작한다.

$$\tau = k \cdot (q_\text{target} - q) + d \cdot (\dot{q}_\text{target} - \dot{q})$$

이를 2차 선형계로 쓰면 ($\dot{q}_\text{target}=0$ 가정):

$$J\ddot{q} + d\dot{q} + kq = k\,q_\text{target}$$

고유 주파수와 감쇠비:

$$\omega_n = \sqrt{\frac{k}{J}}, \qquad \zeta = \frac{d}{2\sqrt{kJ}}$$

역산:

$$k = J\,\omega_n^2, \qquad d = 2\zeta\omega_n J$$

> **J (관절 관성)** 은 URDF `<inertial>` 에서 추출하거나 `--J_kgm2` 옵션으로 지정.
> AX-18A 링크 기준 대략 0.001 ~ 0.005 kg·m².

AX-18A 실제 내부 제어는 **P-only + compliance margin** 이므로
고속 응답에서 선형 PD와 오차가 발생할 수 있다.
Step 응답 fit 결과의 *effective* k, d를 사용하는 것으로 충분하다.

---

## 파일 구성

```
SOLO_ws/scripts/
  sysid_collect.py    # [라즈베리파이] 관절별 step 명령 + 응답 로깅
  sysid_analyze.py    # [데스크탑]    CSV 분석, 모델 fit, 플롯 생성, 파라미터 추천
```

---

## 절차

### 준비

```bash
# 공통 의존성 (라즈베리파이 + 데스크탑)
pip install numpy pyyaml scipy matplotlib
```

라즈베리파이에는 추가로 `dynamixel_sdk` 필요.

---

### Step 1. 데이터 수집 — 라즈베리파이

**안전 조건**
- 로봇을 **공중에 매달거나 다리가 지면에 닿지 않도록** 고정한다.
- `step_deg` 는 ≤ 15° 유지 (기본 10°).
- 전원 차단 버튼을 손 가까이 둔다.

```bash
cd ~/SOLO_ws

# 전체 12관절 순차 테스트 (±10°)
python scripts/sysid_collect.py --config config.yaml --step_deg 10

# 특정 관절만 (먼저 2~3개로 확인할 때)
python scripts/sysid_collect.py --config config.yaml --step_deg 10 --joints 0,1,4
```

**실행 흐름**
1. 전체 관절의 `joint_idx | servo_id | name | 현재 위치` 출력
2. ENTER 키로 시작
3. 각 관절마다:
   - `servo_id` 및 target 각도 출력 (← 미스매치 확인용)
   - `+step_deg` 명령 → 1.5s 기록
   - baseline 복귀 → `-step_deg` 명령 → 1.5s 기록
   - baseline 복귀
4. Ctrl+C → baseline 복귀 후 토크 disable

**출력 예시**
```
[sysid] Baseline pose (joint_idx | servo_id | name):
   [ 0] servo_id= 1  L_HipYaw_Joint            +0.003 rad  (+0.2°)
   [ 4] servo_id= 5  L_Thigh_Joint             +0.012 rad  (+0.7°)
   ...

============================================================
  Joint idx : 4
  Name      : L_Thigh_Joint
  Servo ID  : 5  ← SENDING COMMANDS TO THIS MOTOR
  Range     : [-0.785, 0.785] rad  ([-45.0°, 45.0°])
  Baseline  : +0.012 rad  (+0.7°)
  Target +  : +0.187 rad  (+10.7°)
  Target -  : -0.163 rad  (-9.3°)
============================================================
  [+step] servo 5 → target +0.187 rad (+10.7°)  recording...
  [+step] done.
  [-step] servo 5 → target -0.163 rad (-9.3°)  recording...
  [-step] done.
  saved → logs/sysid/joint04_L_Thigh_Joint.csv
```

**결과물**
```
logs/sysid/
  joint00_L_HipYaw_Joint.csv
  joint01_R_HipYaw_Joint.csv
  ...
  joint11_R_AnkleRoll_Joint.csv
  manifest.json
```

각 CSV 형식:

| direction | t_s   | target_rad | pos_rad |
|-----------|-------|------------|---------|
| pos       | 0.000 | 0.1874     | 0.0123  |
| pos       | 0.010 | 0.1874     | 0.0451  |
| ...       | ...   | ...        | ...     |
| neg       | 0.000 | -0.1634    | 0.0123  |

---

### Step 2. 데이터 전송

```bash
# 라즈베리파이 → 데스크탑
scp -r pi@<ROBOT_IP>:~/SOLO_ws/logs/sysid ~/IsaacLab/SOLO_ws/logs/sysid
```

---

### Step 3. 분석 — 데스크탑

```bash
cd ~/IsaacLab/SOLO_ws

python scripts/sysid_analyze.py \
    --in_dir  logs/sysid \
    --out_dir logs/sysid/plots \
    --k_current 4.5 \
    --d_current 0.45 \
    --J_kgm2 0.002
```

| 옵션 | 설명 |
|------|------|
| `--k_current` | 현재 sim stiffness (비교 baseline) |
| `--d_current` | 현재 sim damping |
| `--J_kgm2`   | 관절 관성 [kg·m²]. URDF 값 사용 권장 |

---

### Step 4. 결과 확인

**플롯 — 관절별** (`plots/joint04_L_Thigh_Joint.png`)

```
┌──────────────── +step ─────────────┐  ┌──────────────── -step ─────────────┐
│  target (----)                     │  │                                     │
│  real   (blue)                     │  │                                     │
│  fit    (red):  ωn=28.3, ζ=0.72   │  │                                     │
│  sim cur.(green): k=4.5, d=0.45   │  │                                     │
└───────────────────────────────────┘  └────────────────────────────────────┘
   rise=63ms, OS=8.1%, BW=4.5Hz           rise=65ms, OS=7.3%, BW=4.5Hz
```

**플롯 — 전체 요약** (`plots/summary.png`): 12관절 4×3 grid

**파라미터 추천** (`logs/sysid/recommendations.txt`)

```
Joint                      ωn (rad/s)     ζ    BW(Hz)   k_rec   d_rec
─────────────────────────────────────────────────────────────────────
J 0 L_HipYaw_Joint          31.2        0.68    5.0     1.951   0.085
J 4 L_Thigh_Joint           28.3        0.72    4.5     1.600   0.081
...
MEDIAN                       29.5        0.70    4.7     1.740   0.083

RECOMMENDED:
  ImplicitActuatorCfg(
      stiffness = 1.740,   # was 4.5
      damping   = 0.0826,  # was 0.45
      ...
  )
```

---

### Step 5. sim 파라미터 업데이트

`source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/dextra_amp_env_cfg.py`:

```python
"legs": ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    stiffness=<k_rec>,   # recommendations.txt 의 MEDIAN 값
    damping=<d_rec>,
    effort_limit=1.8,
    velocity_limit=10.16,
)
```

그 후 **teacher 재학습** 필요 (estimator는 k에 의존).

---

## J(관절 관성) 추출 방법

URDF에서 각 링크의 `<inertia>` 태그 참조:

```bash
grep -A5 "<inertial>" \
  source/isaaclab_assets/.../Dextra_lowerbody.urdf | head -80
```

관절축 방향(z축 기준)의 관성 모멘트 `Izz` 를 사용.
모터 회전자 관성이 있다면 gear ratio² × J_rotor 를 더한다.

---

## 주의사항

| 항목 | 내용 |
|------|------|
| 로봇 자세 | 공중 매달기 필수. 지면 접촉 시 링크 관성이 달라짐 |
| step 크기 | 10° 권장. 15° 이상은 compliance 비선형 영역 진입 가능 |
| J 불확실성 | J를 2배 틀리면 k 추정도 2배 오차. URDF 값 사용 권장 |
| 재현성 | 관절당 ±step 평균 사용. 관절간 차이가 크면 개별 k/d 고려 |
| AX-18A 구조 | P-only + compliance → 고주파(>8Hz)에서 sim PD와 오차 발생 가능 |
