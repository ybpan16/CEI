# CEI

CEI performs tests on paired clone counts (baseline vs post), designed as the joint detection of expanded TCR clones using Mahalanobis distances with permutation and ACAT, under FDR control.

## Installation

```bash
pip install git+https://github.com/ybpan16/cei.git
```

## Run-example

```bash
cei -i ./tests/CLE0083_BP1.xlsx --Bcol "B" --Pcol "P1" --out-joint ./tests/CLE0083_BP1_joint.xlsx
```

