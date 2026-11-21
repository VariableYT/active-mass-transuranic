# Physics audit, active-mass model (paper rev 5k)

## Test 1: EP tail leakage and the detectable window

MICROSCOPE suppression at 710 km: exp(-h/lam) = 10^(-6.05e+05) at lam = 0.51 m. Its Ti-Pt bound says nothing about this model.

| q0 | lam (m) | 5-sigma window in sigma (U pair) | 95 % exclusion window | what caps q_c at the wide edge |
|---|---|---|---|---|
| 0 | 0.02 | 4.75-20.75 | 4.75-25 | Smith (Cu-Pb toward U) |
| 0 | 0.05 | 4.75-13.75 | 4.75-15 | Smith (Cu-Pb toward U) |
| 0 | 0.1 | 4.75-12.5 | 4.75-13.25 | Smith (Cu-Pb toward U) |
| 0 | 0.2 | 4.75-11.75 | 4.75-12.5 | Smith (Cu-Pb toward U) |
| 0 | 0.51 | 4.75-11.5 | 4.75-12.25 | Smith (Cu-Pb toward U) |
| 1e-3 | 0.02 | 4.75-20.75 | 4.75-25 | Smith (Cu-Pb toward U) |
| 1e-3 | 0.05 | 4.75-13.75 | 4.75-14.75 | Smith (Cu-Pb toward U) |
| 1e-3 | 0.1 | 4.75-12 | 4.75-13 | Smith (Cu-Pb toward U) |
| 1e-3 | 0.2 | 4.75-10.75 | 4.75-12 | Smith (Cu-Pb toward U) |
| 1e-3 | 0.51 | none | 4.75-11 | Smith (Cu-Pb toward U) |

Max tail eta at the largest allowed q_c, over sigma 3-40 (all-horizontal ground field, q0 = 1e-3):

| lam | eta Be-Ti | eta Cu-Pb | eta Ti-Pt (if at ground level) | eta Ti-Pt (MICROSCOPE, 710 km) |
|---|---|---|---|---|
| 0.02 | 3.0e-19 | 2.6e-15 | 1.7e-15 | 0 |
| 0.05 | 6.0e-20 | 5.3e-16 | 3.6e-16 | 0 |
| 0.1 | 3.0e-20 | 2.6e-16 | 1.8e-16 | 0 |
| 0.2 | 2.5e-20 | 2.2e-16 | 1.5e-16 | 0 |
| 0.51 | 4.1e-20 | 3.6e-16 | 2.4e-16 | 0 |

## Test 2: self-screening

| body | lam | R (cm) | R/lam | form factor phi | 
|---|---|---|---|---|
| DU 1kg | 0.02 | 2.32 | 1.16 | 1.142 |
| DU 1kg | 0.05 | 2.32 | 0.465 | 1.022 |
| DU 1kg | 0.1 | 2.32 | 0.232 | 1.005 |
| DU 1kg | 0.2 | 2.32 | 0.116 | 1.001 |
| DU 1kg | 0.51 | 2.32 | 0.0455 | 1 |
| DU 10kg | 0.02 | 5.00 | 2.5 | 1.783 |
| DU 10kg | 0.05 | 5.00 | 1 | 1.104 |
| DU 10kg | 0.1 | 5.00 | 0.5 | 1.025 |
| DU 10kg | 0.2 | 5.00 | 0.25 | 1.006 |
| DU 10kg | 0.51 | 5.00 | 0.0981 | 1.001 |
| Th 1kg | 0.02 | 2.73 | 1.37 | 1.199 |
| Th 1kg | 0.05 | 2.73 | 0.546 | 1.03 |
| Th 1kg | 0.1 | 2.73 | 0.273 | 1.007 |
| Th 1kg | 0.2 | 2.73 | 0.137 | 1.002 |
| Th 1kg | 0.51 | 2.73 | 0.0535 | 1 |
| Th 10kg | 0.02 | 5.88 | 2.94 | 2.18 |
| Th 10kg | 0.05 | 5.88 | 1.18 | 1.146 |
| Th 10kg | 0.1 | 5.88 | 0.588 | 1.035 |
| Th 10kg | 0.2 | 5.88 | 0.294 | 1.009 |
| Th 10kg | 0.51 | 5.88 | 0.115 | 1.001 |

Exponent of R in the Yukawa field at a 2 cm gap (3 = volume, 2 = surface; Newton for reference):

- lam 0.02 m: Yukawa 1.04, Newton 1.74
- lam 0.05 m: Yukawa 1.47, Newton 1.74
- lam 0.1 m: Yukawa 1.64, Newton 1.74
- lam 0.2 m: Yukawa 1.71, Newton 1.74
- lam 0.51 m: Yukawa 1.74, Newton 1.74

Nonlinearity eps = 4 pi G rho q^2 lam^2 / c^2 for DU at the q0 bound: 3.4e-28 (lam 0.51 m). Bulk island matter goes nonlinear at q = 1.4e+11; a single A = 298 nucleus at q = 4.7e+18; decay-energy ceiling at width 5 is 2.9e+16.

## Test 3: 3D torque, thermal floor, machining

Thermal ASD 9.2e-16 N m/rtHz (paper pendulum), 1-day sigma 4.4e-18 vs the paper's time-domain 4.6e-18; 30-day sigma 8.1e-19 N m. Quadrature convergence 7.6e-07; sphere vs analytic 5.7e-09.

| source | gap | tau_N (N m) | Newton 3D/point | P floor 95 % (thermal, 30 d) at lam = 0.02 / 0.05 / 0.1 / 0.51 m |
|---|---|---|---|---|
| DU,1kg | 2cm | 2.0e-12 | 0.960 | 2.0e-06 / 1.0e-06 / 8.6e-07 / 8.0e-07 |
| DU,1kg | 3cm | 1.4e-12 | 0.976 | 4.2e-06 / 1.7e-06 / 1.3e-06 / 1.2e-06 |
| DU,1kg | 5cm | 7.3e-13 | 0.989 | 1.7e-05 / 3.8e-06 / 2.7e-06 / 2.2e-06 |
| DU,1kg | 7cm | 4.5e-13 | 0.993 | 6.0e-05 / 7.9e-06 / 4.7e-06 / 3.6e-06 |
| DU,1kg | 10cm | 2.6e-13 | 0.996 | 3.7e-04 / 2.1e-05 / 9.5e-06 / 6.4e-06 |
| DU,3kg | 2cm | 4.1e-12 | 0.937 | 1.2e-06 / 5.4e-07 / 4.4e-07 / 4.0e-07 |
| DU,3kg | 3cm | 2.9e-12 | 0.959 | 2.5e-06 / 8.3e-07 / 6.3e-07 / 5.5e-07 |
| DU,3kg | 5cm | 1.7e-12 | 0.980 | 9.1e-06 / 1.8e-06 / 1.2e-06 / 9.5e-07 |
| DU,3kg | 7cm | 1.1e-12 | 0.988 | 3.1e-05 / 3.5e-06 / 2.0e-06 / 1.5e-06 |
| DU,3kg | 10cm | 6.8e-13 | 0.993 | 1.9e-04 / 8.9e-06 / 3.8e-06 / 2.5e-06 |
| DU,7kg | 2cm | 6.6e-12 | 0.916 | 9.5e-07 / 3.6e-07 / 2.8e-07 / 2.5e-07 |
| DU,7kg | 3cm | 5.1e-12 | 0.942 | 1.8e-06 / 5.2e-07 / 3.8e-07 / 3.2e-07 |
| DU,7kg | 5cm | 3.2e-12 | 0.969 | 6.2e-06 / 1.1e-06 / 6.5e-07 / 5.1e-07 |
| DU,7kg | 7cm | 2.2e-12 | 0.981 | 2.1e-05 / 2.0e-06 / 1.1e-06 / 7.6e-07 |
| DU,7kg | 10cm | 1.4e-12 | 0.989 | 1.2e-04 / 4.9e-06 / 2.0e-06 / 1.2e-06 |
| DU,10kg | 2cm | 8.1e-12 | 0.906 | 8.6e-07 / 3.1e-07 / 2.3e-07 / 2.0e-07 |
| DU,10kg | 3cm | 6.3e-12 | 0.934 | 1.6e-06 / 4.4e-07 / 3.1e-07 / 2.6e-07 |
| DU,10kg | 5cm | 4.1e-12 | 0.963 | 5.4e-06 / 8.7e-07 / 5.2e-07 / 4.0e-07 |
| DU,10kg | 7cm | 2.9e-12 | 0.977 | 1.8e-05 / 1.6e-06 / 8.3e-07 / 5.8e-07 |
| DU,10kg | 10cm | 1.8e-12 | 0.987 | 1.0e-04 / 3.9e-06 / 1.5e-06 / 9.2e-07 |
| Th,7kg | 2cm | 5.3e-12 | 0.903 | 1.4e-06 / 4.8e-07 / 3.5e-07 / 3.1e-07 |
| Th,7kg | 3cm | 4.2e-12 | 0.930 | 2.5e-06 / 6.8e-07 / 4.7e-07 / 3.9e-07 |
| Th,7kg | 5cm | 2.7e-12 | 0.961 | 8.4e-06 / 1.3e-06 / 7.8e-07 / 6.0e-07 |
| Th,7kg | 7cm | 1.9e-12 | 0.975 | 2.7e-05 / 2.5e-06 / 1.2e-06 / 8.6e-07 |
| Th,7kg | 10cm | 1.2e-12 | 0.986 | 1.5e-04 / 5.9e-06 / 2.3e-06 / 1.3e-06 |

Newtonian residual, +-10 um on a, h, x, y, z of source and reference (3 kg DU, 3 cm gap), 95 %: as machined 0.0012, mass-matched 4.8e-04 (placement alone 4.7e-04). Thermal floor there: 5.5e-07. Tolerance needed to reach it: 0.012 um. Paper's flip residual: 3.0e-05.

## Test 4: action

- green_homogeneous: True
- green_flux_at_origin: g
- dV_dr: g*g_b*(m*r + 1)*exp(-m*r)/(4*pi*r**2)
- attractive_like_charges: True
- vector_would_repel: True
- dT_field = g s d^nu phi on shell (all nu): True
- total_conserved: dT_field + dT_matter = g s d phi - g s d phi = 0 (diffeomorphism identity)
- kinetic_sign_positive: True
- no_tachyon_if_m2_positive: True
- g_per_unit_q: 2.7046337271083932e-19
- m_eff_eV_at_lam: {'0.02': 9.866349016919822e-06, '0.05': 3.946539606767929e-06, '0.1': 1.9732698033839643e-06, '0.2': 9.866349016919822e-07, '0.51': 3.8691564772234595e-07}
- perturbative_q_limit_g_lt_1: 3.697358315017134e+18
- qc_decay_ceiling_as_g: {'3.5': 0.0188412803298063, '5': 0.007946007771481906, '7': 0.005894623607779964, '10': 0.005579420125819051}
- LLI_if_coupled_to_J0: {'frac_shift_cmb': 7.608689871485552e-07, 'annual_modulation': 2.450776331654386e-07, 'note': 'coupling to n_B (time component of a 4-vector) gives a frame-dependent charge; use m_X(phi) or phi psi-bar psi'}