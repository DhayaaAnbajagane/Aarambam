from ..utils import camb2input
import argparse, os, numpy as np

def main():

    args = argparse.ArgumentParser()
    args.add_argument('--transfer_path', action = 'store', type = str, required = True, help = "Path to CAMB transfer function")
    args = args.parse_args()

    #Writes e.g. TransferFunction.dat -> TransferFunctionConverted.dat, as an ASCII
    #table since that is what the 2LPT executables read
    root, ext = os.path.splitext(args.transfer_path)
    outpath   = root + 'Converted' + ext
    np.savetxt(outpath, camb2input(args.transfer_path))
    print(f"Wrote {outpath}")
