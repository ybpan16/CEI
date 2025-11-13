# src/CEI/api.py

import argparse
from .method import run_analysis

def parse_args():
    p = argparse.ArgumentParser(
        description="Joint 1D/2D expansion analysis with configurable FDR and file paths."
    )
    p.add_argument("-i", "--input-file", required=True,
                   help="Input Excel file, e.g. ./data_cleaned/CLE0100_BP1.xlsx")
    p.add_argument("--Bcol", default=None,
                   help="Baseline column name.")
    p.add_argument("--Pcol", default=None,
                   help="Post-infection column name.")

    p.add_argument("--alpha-2d", type=float, default=0.10,
                   help="FDR level for 2D parametric test (default: 0.10).")
    p.add_argument("--alpha-joint", type=float, default=0.10,
                   help="FDR level for joint Mahalanobis+permutation (default: 0.10).")
    p.add_argument("--alpha-acat", type=float, default=0.10,
                   help="FDR level for ACAT combination (default: 0.10).")
    p.add_argument("--fdr-method", choices=["bh", "by"], default="bh",
                   help="FDR method: Benjamini–Hochberg (bh) or Benjamini–Yekutieli (by).")

    p.add_argument("--R-perm", type=int, default=400,
                   help="Number of permutations for joint 2D p-values (default: 400).")

    p.add_argument("--out-2d", default=False,
                   help="Output Excel filename for 2D hits (default auto).")
    p.add_argument("--out-joint", default=False,
                   help="Output Excel filename for joint hits (default auto).")
    p.add_argument("--out-acat", default=False,
                   help="Output Excel filename for ACAT hits (default auto).")
    
    p.add_argument("--outer_name_2d", default=None,
                   help="Output Excel filename for 2D hits (default auto).")
    p.add_argument("--out-joint", default=None,
                   help="Output Excel filename for joint hits (default auto).")
    p.add_argument("--out-acat", default=None,
                   help="Output Excel filename for ACAT hits (default auto).")

    return p.parse_args()

def main():
    args = parse_args()
    run_analysis(
        input_file=args.input_file,
        Bcol=args.Bcol,
        Pcol=args.Pcol,
        alpha_2d=args.alpha_2d,
        alpha_joint=args.alpha_joint,
        alpha_acat=args.alpha_acat,
        fdr_method=args.fdr_method,
        R_perm=args.R_perm,
        out_2d=args.out_2d,
        out_joint=args.out_joint,
        out_acat=args.out_acat,
    )