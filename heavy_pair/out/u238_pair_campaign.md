# Stage 1 with a depleted-uranium test body

4000 null and 1000 injection runs per cell, common random numbers (seed 20261118).

## Error budget at 0.5 m (anomaly units)

| Term | Copper test body | Uranium test body |
|---|---|---|
| newtonian | 1.4e-5 | 1.4e-5 |
| ferromagnetic | 4.0e-6 | 4.0e-6 |
| neutron_10pct | 1.0e-13 | 9.4e-14 |
| heat_1pct | 1.5e-10 | 9.2e-11 |
| gamma | 0.0e0 | 0.0e0 |
| paramagnetic | 1.4e-12 | 2.9e-11 |
| beta momentum from the source, no capsule attenuation (not in the paper) | 1.9e-7 | 1.2e-7 |
| total, paper's terms | 1.50e-5 | 1.50e-5 |
| total with the beta upper bound | 1.50e-5 | 1.50e-5 |

Paramagnetic term against the residual field (uranium test body): 5e-05 T: 7.2e-6, 5e-06 T: 7.2e-8, 5e-07 T: 7.2e-10, 1e-07 T: 2.9e-11, 1e-08 T: 2.9e-13. It stays below 1% of the systematic error for fields under 7.2 uT, i.e. a shielding factor of 7 against the Earth's field.
Test-body remanent moment if its iron sits in 1%-remanent precipitates (the paper's worst case): 0.1 ppm Fe 6.5e-10 A m^2, 1 ppm Fe 6.5e-9 A m^2, 10 ppm Fe 6.5e-8 A m^2, 100 ppm Fe 6.5e-7 A m^2. The paper requires below 1e-9 A m^2, which this worst case meets only below 0.15 ppm. The faked anomaly at each iron level, with the source at its own requirement: 0.1 ppm Fe 2.6e-6, 1 ppm Fe 2.6e-5, 10 ppm Fe 2.6e-4, 100 ppm Fe 2.6e-3. The same holds for a copper test body; the requirement is met by measuring the moment (SQUID) and demagnetizing, not by assuming purity.
Charging: a bare uranium surface releases 84 alphas per second (2.7e-17 A), none through a gold coat thicker than ~10 um; the betas of 234mPa, at most 3.7e4 per second (6.0e-15 A), flow out through the 1.2 m tungsten fibre (297 ohm), raising the test body by 1.8e-12 V.

## Reach and what it means

| Range | Days | Reach S | Present bound on P_U | Copper: P_U improved x | Uranium pair: q_U^2 limit | over free fall (1 ppm) | over bulk density | pair beats the P_U route if q0 < |
|---|---|---|---|---|---|---|---|---|
| 0.05 m | 30 | 3.30e-5 | 1.1e-4 | 3.4 | 3.3e-5 | 6e13 | 2e16 | 2.0e-2 (bound 1.1e-2) |
| 0.05 m | 90 | 3.27e-5 | 1.1e-4 | 3.5 | 3.3e-5 | 7e13 | 2e16 | 2.0e-2 (bound 1.1e-2) |
| 0.1 m | 30 | 3.12e-5 | 2.5e-5 | 0.8 | 3.1e-5 | 2e11 | 6e15 | 4.5e-3 (bound 5.0e-3) |
| 0.1 m | 90 | 3.08e-5 | 2.5e-5 | 0.81 | 3.1e-5 | 2e11 | 6e15 | 4.5e-3 (bound 5.0e-3) |
| 0.2 m | 30 | 3.07e-5 | 1.1e-5 | 0.36 | 3.1e-5 | 1e10 | 6e15 | 2.0e-3 (bound 3.3e-3) |
| 0.2 m | 90 | 3.04e-5 | 1.1e-5 | 0.36 | 3.0e-5 | 1e10 | 6e15 | 2.0e-3 (bound 3.3e-3) |
| 0.5 m | 30 | 3.06e-5 | 7.3e-6 | 0.24 | 3.1e-5 | 9e8 | 6e15 | 1.3e-3 (bound 2.7e-3) |
| 0.5 m | 90 | 3.02e-5 | 7.3e-6 | 0.24 | 3.0e-5 | 9e8 | 7e15 | 1.3e-3 (bound 2.7e-3) |
| 1 m | 30 | 3.06e-5 | 6.7e-6 | 0.22 | 3.1e-5 | 3e8 | 6e15 | 1.2e-3 (bound 2.6e-3) |
| 1 m | 90 | 3.02e-5 | 6.7e-6 | 0.22 | 3.0e-5 | 3e8 | 7e15 | 1.2e-3 (bound 2.6e-3) |
| 10 m | 30 | 3.06e-5 | 6.5e-6 | 0.21 | 3.1e-5 | 2e7 | 6e15 | 1.2e-3 (bound 2.6e-3) |
| 10 m | 90 | 3.02e-5 | 6.5e-6 | 0.22 | 3.0e-5 | 2e7 | 7e15 | 1.2e-3 (bound 2.6e-3) |

Time-domain checks (recovered/injected at the five-sigma threshold, scatter over predicted, detection rate):
- Cu 30 d 0.05 m: 1.017, 1.003, 0.540
- Cu 30 d 0.1 m: 1.017, 1.003, 0.540
- Cu 30 d 0.2 m: 1.017, 1.003, 0.540
- Cu 30 d 0.5 m: 1.017, 1.003, 0.540
- Cu 30 d 1 m: 1.017, 1.003, 0.540
- Cu 30 d 10 m: 1.017, 1.003, 0.540
- U 30 d 0.05 m: 1.017, 1.003, 0.540
- U 30 d 0.1 m: 1.017, 1.003, 0.540
- U 30 d 0.2 m: 1.017, 1.003, 0.540
- U 30 d 0.5 m: 1.017, 1.003, 0.540
- U 30 d 1 m: 1.017, 1.003, 0.540
- U 30 d 10 m: 1.017, 1.003, 0.540
- Cu 90 d 0.05 m: 0.994, 0.995, 0.492
- Cu 90 d 0.1 m: 0.994, 0.995, 0.492
- Cu 90 d 0.2 m: 0.994, 0.995, 0.492
- Cu 90 d 0.5 m: 0.994, 0.995, 0.492
- Cu 90 d 1 m: 0.994, 0.995, 0.492
- Cu 90 d 10 m: 0.994, 0.995, 0.492
- U 90 d 0.05 m: 0.994, 0.995, 0.492
- U 90 d 0.1 m: 0.994, 0.995, 0.492
- U 90 d 0.2 m: 0.994, 0.995, 0.492
- U 90 d 0.5 m: 0.994, 0.995, 0.492
- U 90 d 1 m: 0.994, 0.995, 0.492
- U 90 d 10 m: 0.994, 0.995, 0.492
