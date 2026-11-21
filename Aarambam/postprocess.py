"""Post-processing utilities: collate potential dumps, convert transfer functions."""
import numpy as np, os, gc, glob, re, tqdm

__all__ = ['collate_potential', 'find_potential_files', 'camb2input']


def camb2input(path):
    """
    Read a CAMB transfer-function file as the (k, T(k)) table the 2LPT executables expect.

    Parameters
    ----------
    path : str
        CAMB transfer-function file (k in column 0, total matter in column 6).

    Returns
    -------
    ndarray, shape (N, 2)
        k and the matter transfer function. Save it with ``np.savetxt``.
    """
    return np.loadtxt(path, usecols = [0, 6])
    

def collate_potential(OutputDir, seed = None, cleanup = True):
    """
    Combine the potential files of a run (``SavePotentialField 1``) into full arrays.
    One rank writes ``<field>_potential_<seed>``, several ranks one file per slab,
    ``<field>_potential_<seed>_StartX_<x0>_NumX_<nx>``.

    Parameters
    ----------
    OutputDir : str
        Directory the run wrote the potentials into.
    seed : int, optional
        Seed of the run; needed only if ``OutputDir`` holds files of several seeds.
    cleanup : bool, optional
        Delete the per-rank files once both fields were read (default True).

    Returns
    -------
    dict
        Maps ``'Gauss_potential'`` and ``'Nongauss_potential'`` to
        arrays of shape ``(Nmesh, Nmesh, Nmesh)``.
    """

    dtype = np.dtype([('ind', 'i4'), ('pot', 'f8')])
    found = find_potential_files(OutputDir, seed)

    out = {}
    for t in ['Gauss', 'Nongauss']:

        files = sorted((x0, f) for name, x0, f in found if name == t)
        if len(files) == 0:
            raise ValueError(f"No {t}_potential files for seed {seed} in {OutputDir}")

        #Header is (int total_size, int Nmesh, double Box) = 16 bytes
        Nmesh = int(np.fromfile(files[0][1], dtype = np.int32, count = 2)[1])
        arr   = np.zeros(Nmesh**3, dtype = np.float64)
        i     = 0
        #Loop over the slabs (in order of their starting x-index) and concat
        for x0, filename in tqdm.tqdm(files, desc = f"Collating files for {t}_potential"):

            if x0 * Nmesh**2 != i:
                raise ValueError(f"Slab {filename} starts at x = {x0} but the previous slabs end at x = {i // Nmesh**2}")

            tmp = np.memmap(filename, offset = 16, dtype = dtype, mode = 'r')
            arr[i:i+tmp.size] = tmp['pot']

            i += tmp.size
            del tmp; gc.collect()

        if i != Nmesh**3:
            raise ValueError(f"{t}_potential slabs for seed {seed} hold {i} cells, expected Nmesh^3 = {Nmesh**3}")

        out[t + '_potential'] = arr.reshape([Nmesh]*3)

    #Only remove the inputs once everything was read successfully
    if cleanup:
        for x in found: os.remove(x[2])

    return out


def find_potential_files(OutputDir, seed = None):
    """
    List the potential files of one run in ``OutputDir``.

    Parameters
    ----------
    OutputDir : str
        Directory the run wrote the potentials into.
    seed : int, optional
        Seed of the run; needed only if ``OutputDir`` holds files of several seeds.

    Returns
    -------
    list of (str, int, str)
        ``(field, start_x, path)`` per file, where field is
        ``'Gauss'`` or ``'Nongauss'``.
    """

    pattern = re.compile(r'^(Gauss|Nongauss)_potential_(-?\d+)(?:_StartX_(\d+)_NumX_(\d+))?$')

    found = {}
    for f in glob.glob(os.path.join(OutputDir, '*_potential_*')):
        m = pattern.match(os.path.basename(f))
        if m is not None:
            found.setdefault(int(m.group(2)), []).append((m.group(1), int(m.group(3) or 0), f))

    if seed is None:
        if len(found) != 1:
            raise ValueError(f"Found potential files for seeds {sorted(found)} in {OutputDir}. Pass seed = ... to choose one.")
        seed = list(found)[0]
    elif seed not in found:
        raise ValueError(f"No potential files for seed {seed} in {OutputDir}. Found seeds {sorted(found)}.")

    return found[seed]
