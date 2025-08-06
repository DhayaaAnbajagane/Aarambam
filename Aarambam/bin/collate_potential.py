from ..utils import collate_potential
import argparse, numpy as np

def main():
    
    args = argparse.ArgumentParser()
    args.add_argument('--file_dir', action = 'store', type = str, required = True, help = "Directory containing the potential files")
    args = args.parse_args()
    
    res  = collate_potential(OutputDir = args.file_dir)
    for t in res.keys():
        np.save(args.file_dir + f'/{t}.npy', res[t])
        print(f"FINISHED COLLATING {t} FILES TO PATH {args.file_dir + f'/{t}.npy'}")