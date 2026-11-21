# Shell-closure scan (paper Sec. III D)

| family | rows with an island value | blocked by lead at 0.5 m (P_c = 1e4) | smallest k reaching 1e4 |
|---|---|---|---|
| shell binding | 9 | 9 | - |
| shell share | 9 | 0 | 36 |
| shell / liquid-drop barrier | 9 | 0 | 6 |
| gap delta_2n | 12 | 12 | - |
| gap delta_2p | 11 | 11 | - |
| gaps delta_2n + delta_2p | 11 | 11 | - |
| valence product | 3 | 3 | - |

Largest signal / reach over every row and every P_c from 1e4 to 1e10, couplings the masses allow:

| range | uranium limit | lead limit | best | where | best with k <= 20 | where |
|---|---|---|---|---|---|---|
| 50 cm | 7.3e-06 | 7.3e-09 | 0.013 | Cm-248, shell share, FRDM2012, centre 114, P_c 1e+10, k 59.6 | 0.00033 | Pu-239, shell / liquid-drop barrier, FRDM2012, centre 126, P_c 1e+10, k 10.4 |
| 10 cm | 2.5e-05 | 2.5e-08 | 0.035 | Cm-248, shell share, FRDM2012, centre 114, P_c 1e+10, k 57.9 | 0.0011 | Pu-239, shell / liquid-drop barrier, FRDM2012, centre 126, P_c 1e+10, k 10.1 |
| 5 cm | 0.00011 | 1.1e-07 | 0.12 | Cm-248, shell share, FRDM2012, centre 114, P_c 1e+10, k 55.6 | 0.0043 | Pu-239, shell / liquid-drop barrier, FRDM2012, centre 126, P_c 1e+10, k 9.77 |
| 2 cm | 0.004 | 4e-06 | 1.9 | Cm-248, shell share, FRDM2012, centre 114, P_c 1e+10, k 50.4 | 0.11 | Pu-239, shell / liquid-drop barrier, FRDM2012, centre 126, P_c 1e+10, k 8.86 |

Per source, largest signal / reach (all k; k <= 20):

| range | Pu-239 | Pu-242 | Pu-244 | Cm-248 |
|---|---|---|---|---|
| 50 cm | 0.0057; 0.00033 | 5.7e-05; 5.7e-05 | 4.1e-05; 4.1e-05 | 0.013; 8.1e-05 |
| 10 cm | 0.017; 0.0011 | 0.00019; 0.00019 | 0.00014; 0.00014 | 0.035; 0.00025 |
| 5 cm | 0.062; 0.0043 | 0.00084; 0.00084 | 0.00063; 0.00063 | 0.12; 0.001 |
| 2 cm | 1.2; 0.11 | 0.024; 0.024 | 0.018; 0.018 | 1.9; 0.026 |

Cases at or above a source's reach (couplings the masses allow):

- 2 cm, shell share, FRDM2012, centre 114, P_c 1e+10, k 50.4: Pu-239 1.2 x reach
- 2 cm, shell share, FRDM2012, centre 114, P_c 1e+10, k 50.4: Cm-248 1.9 x reach
- 2 cm, shell share, FRDM2012, centre 120, P_c 1e+10, k 49.8: Pu-239 1.1 x reach
- 2 cm, shell share, FRDM2012, centre 120, P_c 1e+10, k 49.8: Cm-248 1.8 x reach
- 2 cm, shell share, FRDM2012, centre 126, P_c 1e+10, k 49.8: Pu-239 1.1 x reach
- 2 cm, shell share, FRDM2012, centre 126, P_c 1e+10, k 49.8: Cm-248 1.8 x reach
