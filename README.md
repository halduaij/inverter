# inverter

This repository contains a small inverter network model implemented in PyTorch.
The simulation can operate in the standard SI unit system or in per‑unit (p.u.)
form. Running in per‑unit scales all voltages, currents and powers by the base
values defined in `PowerSystemNetwork`.

To verify that the per‑unit implementation is consistent with the original
model, set `mode = "per_unit_check"` in `inverter_code.py` and execute the
script:

```bash
python inverter_code.py
```

The program will run a base simulation and a per‑unit simulation of the same
scenario and report the maximum deviation between the two.
