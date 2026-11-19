"""Larger checks that need a quiet node. Not part of the pytest suite.

Usage:
    python tests/large/run_large_tests.py <bin_dir> <work_dir> [--nproc 8] [--force]

<bin_dir> holds 2LPTBasis and 2LPTBinned (an installed package's Aarambam/bin, or a build dir). The script
refuses to start if the 1-minute load average is above 25% of the cores, unless --force is given.

Checks, for basis, resbasis (Pk_mode 1) and binned:
  * Nmesh = 384, Nsample = 256 (the production 3/2 rule), GlassTileFac 4: runs on 1 and on --nproc ranks
    give identical potentials (rank independence at a realistic size)
  * Nmesh = Nsample = 256 on --nproc ranks: NG/Gaussian rms ratio sane, no NaN
Memory: ~4 GB per run at 384^3.
"""
import argparse, glob, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import conftest  # noqa: E402  (registers the Aarambam modules without the compiled extension if needed)
import importlib  # noqa: E402
U = importlib.import_module('Aarambam.utils')

DT  = np.dtype([('ind', 'i4'), ('pot', 'f8')])
EXE = {'basis': '2LPTBasis', 'resbasis': '2LPTBasis', 'binned': '2LPTBinned'}


def potential(d, field):
    files = sorted(f for f in glob.glob(os.path.join(d, field + '_potential_*')) if not f.endswith('.npy'))
    return np.concatenate([np.fromfile(f, dtype=DT, offset=16)['pot'] for f in files])


def run(bindir, method, out, nproc, **kw):
    os.makedirs(out, exist_ok=True)
    d = dict(getattr(U, f'DEFAULT_{method.upper()}_INPUTS'))
    d.update(OutputDir=out, NWriteoutProcesses=1, SavePotentialField=1, **kw)
    cfg = os.path.join(out, 'config')
    open(cfg, 'w').write(getattr(U, f'make_config_{method}')(**d))
    env = dict(os.environ, OMP_NUM_THREADS='1')
    r = subprocess.run(['mpirun', '-np', str(nproc), os.path.join(bindir, EXE[method]), cfg],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit(f'{method} {out} failed:\n{r.stdout[-3000:]}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bin_dir'); p.add_argument('work_dir')
    p.add_argument('--nproc', type=int, default=8); p.add_argument('--force', action='store_true')
    a = p.parse_args()

    load, ncpu = os.getloadavg()[0], os.cpu_count()
    if load > 0.25 * ncpu and not a.force:
        sys.exit(f'load average {load:.1f} on {ncpu} cores: node is busy, not running (use --force to override)')

    extra = {'resbasis': dict(Pk_mode=1)}
    for method in ('basis', 'resbasis', 'binned'):
        kw = dict(Nmesh=384, Nsample=256, Nmax=256, GlassTileFac=4, **extra.get(method, {}))
        d1, dn = (os.path.join(a.work_dir, f'{method}_384_np{n}') for n in (1, a.nproc))
        run(a.bin_dir, method, d1, 1, **kw); run(a.bin_dir, method, dn, a.nproc, **kw)
        for f in ('Gauss', 'Nongauss'):
            same = np.array_equal(potential(d1, f), potential(dn, f))
            print(f'{method:8s} 384/256 {f:8s} np1 == np{a.nproc}: {same}')

        d = os.path.join(a.work_dir, f'{method}_256')
        run(a.bin_dir, method, d, a.nproc, Nmesh=256, Nsample=256, Nmax=256, GlassTileFac=4, **extra.get(method, {}))
        g, ng = potential(d, 'Gauss'), potential(d, 'Nongauss')
        print(f'{method:8s} 256/256 finite: {np.isfinite(ng).all()}  NG/G rms: {np.std(ng - g) / np.std(g):.3e}')


if __name__ == '__main__':
    main()
