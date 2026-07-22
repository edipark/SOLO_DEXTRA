# AX-18A simulation step response

This directory contains the fixed-base simulation used to compare AX-18A
joint responses over a sweep of actuator damping values.

Run from the IsaacLab repository root:

```bash
./isaaclab.sh -p \
  source/isaaclab_tasks/isaaclab_tasks/direct/SOLO_DEXTRA/sysid/run_sim_step_response.py \
  --headless \
  --joint-name L_Thigh_Joint \
  --step-deg 5 \
  --torque-limit-ratio 0.3
```

Each run creates its own timestamped directory under
`logs/ax18a_sysid/sim/` containing the command and response CSV, a JSON
summary, the exact run configuration, and a PNG plot.
