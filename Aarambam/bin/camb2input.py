from ..utils import camb2input
import argparse, numpy as np

def main():
    
    args = argparse.ArgumentParser()
    args.add_argument('--transfer_path', action = 'store', type = str, required = True, help = "Path to CAMB transfer function")
    args = args.parse_args()
    
    split   = args.transer_path.split('.')
    outpath = sum(split[-1]) + 'Converted.' + split[-1]
    np.save(outpath, camb2input(args.transfer_path))