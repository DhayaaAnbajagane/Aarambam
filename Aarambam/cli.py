"""Command-line interface: ``aarambam <command>``, plus the older ``Aarambam-*`` script names as aliases."""
import argparse, os, sys
from importlib.resources import files

#Compiled 2LPT executables (installed into Aarambam/bin by CMake). ResBasis is the modal (Basis)
#executable with the P(k) oscillation enabled through the config (WhichSpectrum, A_res, w_res, phi_res).
BINARIES = {'basis': '2LPTBasis', 'resbasis': '2LPTBasis', 'binned': '2LPTBinned'}


def run_2lpt(method, args):
    """Replace this process with the compiled 2LPT executable for ``method``."""
    exe = str(files("Aarambam") / "bin" / BINARIES[method])
    if not os.path.isfile(exe):
        sys.exit(f"Aarambam: the compiled 2LPT executable {exe} was not found. It is built when Aarambam is "
                 f"installed with pip and FFTW2/GSL/MPI are available; see the installation docs.")
    os.execv(exe, [exe] + list(args))


def write_example_config(method, config_path):
    from . import config
    with open(config_path, 'w') as f:
        f.write(getattr(config, f'make_example_config_{method}')())
    print("FINISHED WRITING CONFIG TO PATH:", config_path)


def collate(file_dir, seed = None):
    import numpy as np
    from .postprocess import collate_potential, find_potential_files

    #Keep the per-rank files until the collated arrays are safely on disk
    files_ = find_potential_files(OutputDir = file_dir, seed = seed)
    res    = collate_potential(OutputDir = file_dir, seed = seed, cleanup = False)
    for t in res.keys():
        np.save(file_dir + f'/{t}.npy', res[t])
        print(f"FINISHED COLLATING {t} FILES TO PATH {file_dir + f'/{t}.npy'}")

    for _, _, f in files_: os.remove(f)


def camb2input(transfer_path):
    import numpy as np
    from .postprocess import camb2input as convert

    #Writes e.g. TransferFunction.dat -> TransferFunctionConverted.dat, as an ASCII
    #table since that is what the 2LPT executables read
    root, ext = os.path.splitext(transfer_path)
    outpath   = root + 'Converted' + ext
    np.savetxt(outpath, convert(transfer_path))
    print(f"Wrote {outpath}")


def _parser():
    p   = argparse.ArgumentParser(prog = 'aarambam', description = 'Aarambam: non-Gaussian initial conditions.')
    sub = p.add_subparsers(dest = 'command', required = True)

    q = sub.add_parser('2lpt', help = 'Run a 2LPT executable (use with mpirun).')
    q.add_argument('method', choices = sorted(BINARIES))
    q.add_argument('config', help = 'Path to the LPT config file')

    q = sub.add_parser('example-config', help = 'Write an example LPT config.')
    q.add_argument('method', choices = sorted(BINARIES))
    q.add_argument('--config_path', required = True, help = 'Path to put the example LPT config')

    q = sub.add_parser('collate', help = 'Combine per-rank potential files into .npy arrays.')
    q.add_argument('--file_dir', required = True, help = 'Directory containing the potential files')
    q.add_argument('--seed', type = int, default = None, help = 'Seed to collate, if the directory holds several runs')

    q = sub.add_parser('camb2input', help = 'Convert a CAMB transfer function into the 2LPT input format.')
    q.add_argument('--transfer_path', required = True, help = 'Path to CAMB transfer function')

    sub.add_parser('version', help = 'Print the Aarambam version.')
    return p


def main(argv = None):
    args = _parser().parse_args(argv)
    if args.command == '2lpt':
        run_2lpt(args.method, [args.config])
    elif args.command == 'example-config':
        write_example_config(args.method, args.config_path)
    elif args.command == 'collate':
        collate(args.file_dir, args.seed)
    elif args.command == 'camb2input':
        camb2input(args.transfer_path)
    elif args.command == 'version':
        from . import __version__
        print(__version__)


#------------------------------------------------------------------------------------------
#Aliases for the older Aarambam-* script names (pyproject [project.scripts])
#------------------------------------------------------------------------------------------

def lpt_basis():    run_2lpt('basis',    sys.argv[1:])
def lpt_resbasis(): run_2lpt('resbasis', sys.argv[1:])
def lpt_binned():   run_2lpt('binned',   sys.argv[1:])

def _example(method):
    p = argparse.ArgumentParser(description = f'Write an example {method} LPT config.')
    p.add_argument('--config_path', action = 'store', type = str, required = True, help = "Path to put the example LPT config")
    write_example_config(method, p.parse_args().config_path)

def example_basis():    _example('basis')
def example_resbasis(): _example('resbasis')
def example_binned():   _example('binned')

def collate_main():
    p = argparse.ArgumentParser(description = 'Combine per-rank potential files into .npy arrays.')
    p.add_argument('--file_dir', action = 'store', type = str, required = True, help = "Directory containing the potential files")
    p.add_argument('--seed',     action = 'store', type = int, default = None, help = "Seed to collate, if the directory holds several runs")
    a = p.parse_args()
    collate(a.file_dir, a.seed)

def camb2input_main():
    p = argparse.ArgumentParser(description = 'Convert a CAMB transfer function into the 2LPT input format.')
    p.add_argument('--transfer_path', action = 'store', type = str, required = True, help = "Path to CAMB transfer function")
    camb2input(p.parse_args().transfer_path)
