"""Post-processing utilities: collate potential dumps, convert transfer functions."""
import numpy as np, os, gc, glob, re, tqdm

__all__ = ['collate_potential', 'find_potential_files', 'camb2input']


def camb2input(path):
    """
    Simple util to convert a camb-formatted T(k) file into
    the format expected by 2LPTIc (and therefore by Aarambam).

    Parameters
    ----------
    path : str
        Path to the T(k) file from CAMB

    Returns
    -------
    numpy array: (N, 2)
        2D array where first column is k-values and
        2nd column is the matter transfer function.

    """
    return np.loadtxt(path, usecols = [0, 6])
    

def collate_potential(OutputDir, seed = None, cleanup = True):
    """
    Utility to combine the potential grid outputs from Aarambam
    (written when ``SavePotentialField`` is set) into single
    numpy arrays. A run on one MPI rank writes one file per field,
    ``<field>_potential_<seed>``; a run on several ranks writes one
    file per slab, ``<field>_potential_<seed>_StartX_<x0>_NumX_<nx>``.

    Parameters
    ----------
    OutputDir : str
        The directory that these potential files were
        written into
    seed : int, optional
        Seed of the run to collate. Required only if ``OutputDir``
        holds potential files from more than one seed.
    cleanup : bool, optional
        If True (default), delete the per-rank files once all
        fields have been read.

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
        The directory that the potential files were written into
    seed : int, optional
        Seed of the run. Required only if ``OutputDir`` holds
        potential files from more than one seed.

    Returns
    -------
    list of (str, int, str)
        ``(field, start_x, path)`` per file, where field is
        ``'Gauss'`` or ``'Nongauss'``.
    """

    pattern = re.compile(r'^(Gauss|Nongauss)_potential_(\d+)(?:_StartX_(\d+)_NumX_(\d+))?$')

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
