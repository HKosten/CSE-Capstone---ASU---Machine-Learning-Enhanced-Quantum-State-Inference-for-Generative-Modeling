# QCBM experiment results

All metrics compare the model with the TRUE distribution, which the model never saw.
MMD, TVD, KL: lower is better. Fidelity, Validity, Coverage: higher is better.
Validity only applies to Bars and Stripes (the only dataset with invalid bitstrings).

## Trained on 1000 samples

| dataset | model | MMD | TVD | KL | Fidelity | Validity | Coverage |
|---|---|---|---|---|---|---|---|
| bas | QCBM-strong | 0.0060 | 0.3754 | 0.4827 | 0.6209 | 0.6246 | 1.0000 |
| bas | QCBM-hea | 0.0110 | 0.5218 | 0.7633 | 0.4724 | 0.4782 | 1.0000 |
| bas | QCBM-iqp | 0.0215 | 0.6129 | 2.0383 | 0.2733 | 0.3871 | 1.0000 |
| bas | counting | 0.0006 | 0.0390 | 0.0055 | 0.9973 | 1.0000 | 1.0000 |
| markov | QCBM-strong | 0.0010 | 0.1516 | 0.1346 | 0.9424 | — | 0.7656 |
| markov | QCBM-hea | 0.0011 | 0.1691 | 0.1619 | 0.9265 | — | 0.8438 |
| markov | QCBM-iqp | 0.0086 | 0.2939 | 0.3103 | 0.8585 | — | 0.7969 |
| markov | counting | 0.0010 | 0.1580 | 1.0255 | 0.9125 | — | 0.6562 |
| gaussian | QCBM-strong | 0.0007 | 0.1609 | 0.1226 | 0.9375 | — | 0.7930 |
| gaussian | QCBM-hea | 0.0007 | 0.1719 | 0.1569 | 0.9194 | — | 0.8008 |
| gaussian | QCBM-iqp | 0.0019 | 0.3231 | 0.4920 | 0.7544 | — | 0.8789 |
| gaussian | counting | 0.0007 | 0.1527 | 0.6273 | 0.9349 | — | 0.5977 |
| quantum | QCBM-strong | 0.0006 | 0.1511 | 0.1457 | 0.9247 | — | 0.8047 |
| quantum | QCBM-hea | 0.0007 | 0.1806 | 0.1764 | 0.9032 | — | 0.8320 |
| quantum | QCBM-iqp | 0.0026 | 0.3315 | 0.5098 | 0.7511 | — | 0.8750 |
| quantum | counting | 0.0005 | 0.1311 | 1.0046 | 0.9228 | — | 0.5547 |

## Trained on 100 samples

| dataset | model | MMD | TVD | KL | Fidelity | Validity | Coverage |
|---|---|---|---|---|---|---|---|
| bas | QCBM-strong | 0.0113 | 0.4058 | 0.6225 | 0.5817 | 0.6261 | 1.0000 |
| bas | counting | 0.0062 | 0.1329 | 0.0531 | 0.9737 | 1.0000 | 1.0000 |
| markov | QCBM-strong | 0.0056 | 0.2859 | 0.3845 | 0.8455 | — | 0.7305 |
| markov | counting | 0.0066 | 0.3668 | 6.5768 | 0.6255 | — | 0.2266 |
| gaussian | QCBM-strong | 0.0042 | 0.3625 | 0.6947 | 0.7532 | — | 0.7070 |
| gaussian | counting | 0.0049 | 0.4266 | 8.8374 | 0.5525 | — | 0.2891 |
| quantum | QCBM-strong | 0.0054 | 0.3362 | 0.5129 | 0.7886 | — | 0.7188 |
| quantum | counting | 0.0060 | 0.3776 | 5.6926 | 0.6370 | — | 0.2148 |

## Run details

| dataset | ansatz | parameters | steps | seconds |
|---|---|---|---|---|
| bas | strong | 270 | 400 | 57.9 |
| bas | hea | 180 | 400 | 43.2 |
| bas | iqp | 45 | 400 | 12.5 |
| markov | strong | 240 | 400 | 49.2 |
| markov | hea | 160 | 400 | 36.9 |
| markov | iqp | 36 | 400 | 10.2 |
| gaussian | strong | 240 | 400 | 48.7 |
| gaussian | hea | 160 | 400 | 36.7 |
| gaussian | iqp | 36 | 400 | 9.7 |
| quantum | strong | 240 | 400 | 48.6 |
| quantum | hea | 160 | 400 | 36.5 |
| quantum | iqp | 36 | 400 | 9.6 |
