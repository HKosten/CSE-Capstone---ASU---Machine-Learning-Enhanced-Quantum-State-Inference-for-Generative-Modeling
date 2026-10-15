# QCBM stability + sample size sweep

circuit: strong (10 layers), 40 steps max, 2 seeds per setting.
Each seed = new training samples AND new starting angles.
Numbers are mean ± std over seeds. All metrics are vs the TRUE distribution.
"QCBM wins" = in how many seeds the QCBM beat plain counting on that metric.
smoothed = counting + 0.5 pseudo-count on every bitstring.

## Where does counting catch up? (mean TVD)

- **bas**: Counting has lower mean TVD than the QCBM at every size tested.
- **markov**: Counting catches up at n = 1000 and stays better for every bigger size.
- **gaussian**: Counting catches up at n = 1000 and stays better for every bigger size.
- **quantum**: Counting catches up at n = 1000 and stays better for every bigger size.

## bas

| n | metric | QCBM | counting | smoothed | QCBM wins |
|---|---|---|---|---|---|
| 50 | tvd | 0.6299 ± 0.0366 | 0.1900 ± 0.0141 | 0.8137 ± 0.0000 | 0/2 |
| 50 | mmd | 0.0212 ± 0.0013 | 0.0113 ± 0.0034 | 0.0269 ± 0.0001 | 0/2 |
| 50 | kl | 1.6785 ± 0.0176 | 0.9685 ± 1.1947 | 1.8044 ± 0.0521 | 1/2 |
| 50 | coverage | 0.6786 ± 0.0505 | 0.9643 ± 0.0505 | 0.5714 ± 0.1010 | 0/2 |
| 50 | validity | 0.3902 ± 0.0147 | 1.0000 ± 0.0000 | 0.1863 ± 0.0000 | 0/2 |
| 1000 | tvd | 0.5964 ± 0.0043 | 0.0394 ± 0.0006 | 0.1982 ± 0.0000 | 0/2 |
| 1000 | mmd | 0.0135 ± 0.0001 | 0.0006 ± 0.0001 | 0.0020 ± 0.0001 | 0/2 |
| 1000 | kl | 0.9470 ± 0.0008 | 0.0055 ± 0.0001 | 0.2264 ± 0.0001 | 0/2 |
| 1000 | coverage | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0/2 |
| 1000 | validity | 0.4036 ± 0.0043 | 1.0000 ± 0.0000 | 0.8018 ± 0.0000 | 0/2 |

## markov

| n | metric | QCBM | counting | smoothed | QCBM wins |
|---|---|---|---|---|---|
| 50 | tvd | 0.3862 ± 0.0196 | 0.5330 ± 0.0222 | 0.4125 ± 0.0047 | 2/2 |
| 50 | mmd | 0.0083 ± 0.0016 | 0.0117 ± 0.0019 | 0.0151 ± 0.0003 | 2/2 |
| 50 | kl | 0.6919 ± 0.0706 | 10.8438 ± 0.5192 | 0.5642 ± 0.0249 | 2/2 |
| 50 | coverage | 0.1426 ± 0.0138 | 0.1465 ± 0.0028 | 0.1719 ± 0.0110 | 1/2 |
| 1000 | tvd | 0.2382 ± 0.0279 | 0.1499 ± 0.0114 | 0.1526 ± 0.0105 | 0/2 |
| 1000 | mmd | 0.0014 ± 0.0006 | 0.0007 ± 0.0004 | 0.0009 ± 0.0004 | 0/2 |
| 1000 | kl | 0.3114 ± 0.1316 | 0.9867 ± 0.0550 | 0.0932 ± 0.0116 | 2/2 |
| 1000 | coverage | 0.7656 ± 0.0166 | 0.6562 ± 0.0000 | 0.7500 ± 0.0055 | 2/2 |

## gaussian

| n | metric | QCBM | counting | smoothed | QCBM wins |
|---|---|---|---|---|---|
| 50 | tvd | 0.4896 ± 0.0157 | 0.6414 ± 0.0034 | 0.4242 ± 0.0089 | 2/2 |
| 50 | mmd | 0.0126 ± 0.0004 | 0.0157 ± 0.0003 | 0.0057 ± 0.0002 | 2/2 |
| 50 | kl | 1.0979 ± 0.1076 | 14.2272 ± 0.0644 | 0.5971 ± 0.0076 | 2/2 |
| 50 | coverage | 0.1406 ± 0.0110 | 0.1660 ± 0.0083 | 0.1641 ± 0.0055 | 0/2 |
| 1000 | tvd | 0.2349 ± 0.0133 | 0.1523 ± 0.0004 | 0.1518 ± 0.0004 | 0/2 |
| 1000 | mmd | 0.0012 ± 0.0001 | 0.0007 ± 0.0000 | 0.0006 ± 0.0000 | 0/2 |
| 1000 | kl | 0.3243 ± 0.0056 | 0.6216 ± 0.0081 | 0.1059 ± 0.0108 | 2/2 |
| 1000 | coverage | 0.7227 ± 0.0276 | 0.5898 ± 0.0110 | 0.6875 ± 0.0055 | 2/2 |

## quantum

| n | metric | QCBM | counting | smoothed | QCBM wins |
|---|---|---|---|---|---|
| 50 | tvd | 0.4313 ± 0.0023 | 0.5004 ± 0.0003 | 0.4851 ± 0.0144 | 2/2 |
| 50 | mmd | 0.0096 ± 0.0005 | 0.0117 ± 0.0004 | 0.0075 ± 0.0004 | 2/2 |
| 50 | kl | 0.8940 ± 0.1215 | 9.1836 ± 0.2577 | 0.7194 ± 0.0221 | 2/2 |
| 50 | coverage | 0.1270 ± 0.0083 | 0.1387 ± 0.0028 | 0.1680 ± 0.0110 | 0/2 |
| 1000 | tvd | 0.2778 ± 0.0204 | 0.1371 ± 0.0085 | 0.1416 ± 0.0032 | 0/2 |
| 1000 | mmd | 0.0014 ± 0.0004 | 0.0007 ± 0.0002 | 0.0007 ± 0.0002 | 0/2 |
| 1000 | kl | 0.3928 ± 0.0122 | 0.9099 ± 0.1339 | 0.0891 ± 0.0027 | 2/2 |
| 1000 | coverage | 0.7266 ± 0.0442 | 0.5703 ± 0.0221 | 0.6426 ± 0.0083 | 2/2 |

## Run details

16 runs, 1.2 min of training total, 4.4s per run on average.
