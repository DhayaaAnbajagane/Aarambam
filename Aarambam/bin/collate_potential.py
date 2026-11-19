from ..utils import collate_potential, find_potential_files
import argparse, os, numpy as np

def main():

    args = argparse.ArgumentParser()
    args.add_argument('--file_dir', action = 'store', type = str, required = True, help = "Directory containing the potential files")
    args.add_argument('--seed',     action = 'store', type = int, default = None, help = "Seed to collate, if the directory holds several runs")
    args = args.parse_args()

    #Keep the per-rank files until the collated arrays are safely on disk
    files = find_potential_files(OutputDir = args.file_dir, seed = args.seed)
    res   = collate_potential(OutputDir = args.file_dir, seed = args.seed, cleanup = False)
    for t in res.keys():
        np.save(args.file_dir + f'/{t}.npy', res[t])
        print(f"FINISHED COLLATING {t} FILES TO PATH {args.file_dir + f'/{t}.npy'}")

    for _, _, f in files: os.remove(f)
