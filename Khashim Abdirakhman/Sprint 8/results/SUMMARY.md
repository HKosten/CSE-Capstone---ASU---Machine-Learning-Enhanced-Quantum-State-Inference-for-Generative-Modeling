# QCBM experiment results (sprint 8 rerun)

All metrics compare the model with the TRUE distribution, which the model never saw.
MMD, TVD, KL, Test MMD: lower is better. Fidelity, Validity, Coverage: higher is better.
Validity only applies to Bars and Stripes (the only dataset with invalid bitstrings).
Coverage: every model gets the same number of draws as there were training samples.
smoothed = counting + 0.5 pseudo-count on every bitstring.

data seed 0, init seed 42

## Trained on 1000 samples

| dataset | model | MMD | TVD | KL | Fidelity | Validity | Coverage | Test MMD |
|---|---|---|---|---|---|---|---|---|
| bas | QCBM-strong | 0.0136 | 0.5908 | 1.0608 | 0.3869 | 0.4092 | 1.0000 | 0.0139 |
| bas | QCBM-hea | 0.0182 | 0.6719 | 1.2277 | 0.3110 | 0.3281 | 1.0000 | 0.0188 |
| bas | QCBM-iqp | 0.0380 | 0.9713 | 4.3051 | 0.0213 | 0.0287 | 0.7143 | 0.0377 |
| bas | counting | 0.0006 | 0.0390 | 0.0055 | 0.9973 | 1.0000 | 1.0000 | 0.0006 |
| bas | smoothed | 0.0020 | 0.1982 | 0.2263 | 0.7996 | 0.8018 | 1.0000 | 0.0025 |
| markov | QCBM-strong | 0.0015 | 0.2346 | 0.2689 | 0.8777 | — | 0.6953 | 0.0011 |
| markov | QCBM-hea | 0.0018 | 0.2418 | 0.3002 | 0.8613 | — | 0.7148 | 0.0015 |
| markov | QCBM-iqp | 0.0287 | 0.6277 | 1.6929 | 0.4580 | — | 0.8125 | 0.0289 |
| markov | counting | 0.0010 | 0.1580 | 1.0255 | 0.9125 | — | 0.6562 | 0.0006 |
| markov | smoothed | 0.0013 | 0.1600 | 0.1014 | 0.9489 | — | 0.7617 | 0.0004 |
| gaussian | QCBM-strong | 0.0011 | 0.2233 | 0.3016 | 0.8604 | — | 0.7266 | 0.0005 |
| gaussian | QCBM-hea | 0.0017 | 0.2930 | 0.4619 | 0.7967 | — | 0.7617 | 0.0013 |
| gaussian | QCBM-iqp | 0.0093 | 0.5602 | 1.1600 | 0.4937 | — | 0.7930 | 0.0087 |
| gaussian | counting | 0.0007 | 0.1527 | 0.6273 | 0.9349 | — | 0.5977 | 0.0008 |
| gaussian | smoothed | 0.0006 | 0.1520 | 0.1135 | 0.9386 | — | 0.7070 | 0.0005 |
| quantum | QCBM-strong | 0.0013 | 0.2955 | 0.4388 | 0.7908 | — | 0.7031 | 0.0012 |
| quantum | QCBM-hea | 0.0014 | 0.2948 | 0.4137 | 0.7961 | — | 0.7656 | 0.0012 |
| quantum | QCBM-iqp | 0.0124 | 0.6483 | 1.4972 | 0.4239 | — | 0.8164 | 0.0121 |
| quantum | counting | 0.0005 | 0.1311 | 1.0046 | 0.9228 | — | 0.5547 | 0.0005 |
| quantum | smoothed | 0.0006 | 0.1393 | 0.0910 | 0.9516 | — | 0.6641 | 0.0004 |

## Trained on 100 samples

| dataset | model | MMD | TVD | KL | Fidelity | Validity | Coverage | Test MMD |
|---|---|---|---|---|---|---|---|---|
| bas | QCBM-strong | 0.0200 | 0.6239 | 1.4045 | 0.3239 | 0.4001 | 0.7857 | 0.0215 |
| bas | counting | 0.0062 | 0.1329 | 0.0531 | 0.9737 | 1.0000 | 1.0000 | 0.0057 |
| bas | smoothed | 0.0200 | 0.6994 | 1.2483 | 0.2937 | 0.3006 | 0.8571 | 0.0196 |
| markov | QCBM-strong | 0.0062 | 0.3257 | 0.4680 | 0.8147 | — | 0.2070 | 0.0047 |
| markov | counting | 0.0066 | 0.3668 | 6.5768 | 0.6255 | — | 0.2266 | 0.0053 |
| markov | smoothed | 0.0065 | 0.3212 | 0.3271 | 0.8241 | — | 0.2656 | 0.0059 |
| gaussian | QCBM-strong | 0.0041 | 0.3753 | 0.6313 | 0.7430 | — | 0.2539 | 0.0044 |
| gaussian | counting | 0.0049 | 0.4266 | 8.8374 | 0.5525 | — | 0.2891 | 0.0042 |
| gaussian | smoothed | 0.0038 | 0.3533 | 0.4891 | 0.7328 | — | 0.3086 | 0.0032 |
| quantum | QCBM-strong | 0.0057 | 0.3745 | 0.5788 | 0.7477 | — | 0.2305 | 0.0043 |
| quantum | counting | 0.0060 | 0.3776 | 5.6926 | 0.6370 | — | 0.2148 | 0.0045 |
| quantum | smoothed | 0.0049 | 0.4038 | 0.4729 | 0.7470 | — | 0.2734 | 0.0035 |

## Run details

| dataset | ansatz | parameters | steps | seconds |
|---|---|---|---|---|
| bas | strong | 270 | 40 | 4.8 |
| bas | hea | 180 | 40 | 4.0 |
| bas | iqp | 45 | 40 | 1.0 |
| markov | strong | 240 | 40 | 4.2 |
| markov | hea | 160 | 40 | 3.5 |
| markov | iqp | 36 | 40 | 0.8 |
| gaussian | strong | 240 | 40 | 3.9 |
| gaussian | hea | 160 | 40 | 3.4 |
| gaussian | iqp | 36 | 40 | 1.0 |
| quantum | strong | 240 | 40 | 4.0 |
| quantum | hea | 160 | 40 | 3.3 |
| quantum | iqp | 36 | 40 | 0.8 |
