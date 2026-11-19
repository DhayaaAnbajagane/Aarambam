<p>&nbsp;</p>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/LOGO_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/source/LOGO_light.png">
  <img alt="Logo" src="docs/source/LOGO_dark.png" title="Logo">
</picture>
<p>&nbsp;</p>
<p>&nbsp;</p>

[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/readthedocs/aarambam?color=blue)](https://aarambam.readthedocs.io/en/latest)

## Overview

`Aarambam` (_pronounced "Aah-rum-bum", named after the Tamil word for beginnings_) is a codebase for generating initial conditions (ICs) corresponding to arbitrary bispectrum templates. 

It provides an end-to-end pipeline for generating the ICs of a N-body simulation given some analytic bispectrum template. The novelty in the codebase is performing a decomposition of arbitrary bispectra into separable functions, and then subsequently generating non-Gaussian initial conditions corresponding to these functions. `Aarambam` provides multiple ways of decomposing the bispectrum, each method differing in the regime it is most applicable. The methods have been introduced over a series of different papers. See `Attribution` section below for the requested citations if you use this code. Happy simulating!

(The first two symbols of the logo are from the Tamil script. You can do a process of elimination using the English letters to guess what those symbols' sounds are ;) And thanks to [Emily Martsen](https://emartsen.github.io/) for advice on the logo design!)

## Documentation

I list a few examples below to get you started. More documentation is available [at readthedocs here](https://aarambam.readthedocs.io/en/latest/). Please contact me if you have any questions!

## Environment
The python part of the code has a few dependencies --- namely `numpy`, `scipy` (>= 1.12), `tqdm`, `joblib` and `threadpoolctl`. The last one is more non-standard but is very helpful in managing oversubscription. I promise it is easy to install! `pip install` pulls these in automatically; `cython` is needed only at build time (and is also fetched automatically by `pip`). The C-level code has its own dependencies, I list them below in the Installation instructions. You can install all of these from conda-forge:

```bash
conda create -p {YOUR_ENV_PATH}
conda activate {YOUR_ENV_PATH}
conda install numpy scipy tqdm joblib cython threadpoolctl -c conda-forge
```

## Installation

It is recommended to install the code from source via pip install. This can be done as

```bash
pip install git+https://github.com/DhayaaAnbajagane/Aarambam.git
```

or you can clone the repo yourself first

```bash
git clone https://github.com/DhayaaAnbajagane/Aarambam.git
cd Aarambam
pip install .
```

There are a number of dependencies, particularly due to the 2LPT (2nd order Lagrangian Perturbation Theory) method provided in this package, namely `MPI`, `FFTW2` (NOT `FFTW3`. Using `FFTW2` a legacy dependence from `2LPTIC` days), `GSL` and `OpenMP` (for the Cython extension). The install will generally do its best to find the right compiled objects for these dependencies. `FFTW2` is never searched for automatically, so you must ALWAYS set the following environment variable PRIOR to installation

```bash
export Aarambam_FFTW2_ROOT=/path/to/fftw-2.1.5/build   # the --prefix used for FFTW2; contains include/dfftw.h and lib/
```

If the install is not finding your correct installs of C-compilers, GSL etc. then you can explicitly specify paths with the variables below. On most clusters you will need at least the GSL and MPI ones (otherwise the MPI library is searched for on `LD_LIBRARY_PATH`/`LIBRARY_PATH` and the system paths, and GSL only in `GSL_DIR`/`GSL_ROOT` and the system paths).

```bash
export Aarambam_CC=/path/to/mpicc               # default: the first mpicc on PATH
export Aarambam_GSL_ROOT=/path/to/gsl           # contains include/gsl/ and lib/ (or lib64/)
export Aarambam_OPENMPI_ROOT=/path/to/openmpi   # contains lib/libmpi.so
```

If your computing cluster does not provide an install of FFTW2 (most do not since it is old...) then you can install it via the following steps:

```bash
wget https://www.fftw.org/fftw-2.1.5.tar.gz
tar -xvzf fftw-2.1.5.tar.gz
cd fftw-2.1.5; mkdir build;
./configure --prefix=${PWD}/build --enable-mpi --enable-type-prefix CC={ADD_PATH_TO_MPICC_EXECUTABLE}
make
make install
```
You may be able to skip the `CC=` addition if your computing cluster already supplied `mpicc` in an easily findable way. Both flags are required: `--enable-mpi` builds the MPI-parallel FFTs, and `--enable-type-prefix` gives the double-precision libraries and headers their `d` prefix (`drfftw_mpi.h`, `libdrfftw_mpi`, ...), which is what Aarambam includes and links against. The above `make` command will take a while to run (FFTW2 has a lot of libraries to install). FFTW2 is a hard requirement as this piece of LPT code originates from the 2LPTic code, which was written against the FFTW2 MPI interface (FFTW 2.1.5, from 2003, is the last FFTW2 release).

If you have trouble installing this, please contact Dhayaa (details below). I've tested installation at multiple computing clusters now, but there will always be edge-cases:)

## Example

We provide examples of how to use this code in [this notebook](examples/BinnedBasisDecomposition.ipynb) and [this python script](examples/MakeICsBinned.py). See other notebooks/scripts in those directories for the other execution modes. I copy a snippet of the script below (you will need to fill in other pieces. Look at the scripts for details).

```python
import Aarambam as Am
import numpy as np, subprocess as sp, os

#This is the main code needed for decomposing a given template. Swap "ScalarI" with
#other options to decompose more models. It is very easy to implement your own :)
#If you pass "outdir" then this will also write the coefficient tables in the right
#format into those directories. Passing Binned_Nk selects the binned decomposition
#(pass N_modes, ModeTol and MaxModeCount instead for the Legendre one).
Unit   = Am.utils.Decomposer(n_s = n_s, Lbox = Lbox, Nmax = Nmax, Binned_Nk = 20)
coeffL = Unit.go(Am.models.Local,   Am.binned.BasicBinnedDecompose, outdir = outdir)
coeffS = Unit.go(Am.models.ScalarI, Am.binned.BasicBinnedDecompose, mass = 1, outdir = outdir) #This will overwrite the previous call

#Use the helper function to make the config in the right format and write it to dir.
#binned_kernel_inputs reads the tables written by Unit.go(..., outdir = outdir) and returns
#their paths and the matching C filter codes. Override the other defaults as needed.
inputs = dict(Am.utils.DEFAULT_BINNED_INPUTS)
inputs.update(Am.utils.binned_kernel_inputs(outdir))
inputs.update(Nmesh = 384, Nsample = 256, Nmax = 256, Lbox = Lbox, OutputDir = outdir, Fnl = 100)
CONFIG = Am.utils.make_config_binned(**inputs)
with open(outdir + '/LPTconfig', 'w') as f: f.write(CONFIG)

#Doing a pip install will provide you with executables in your env.
#'aarambam 2lpt binned' runs the executable that reads the binned tables written above
#('aarambam 2lpt basis' / 'resbasis' read the Legendre ones). We run it with
#subprocess just for simplicity. You could also run it from cmd line
sp.run(f"mpirun -np {Nprocs} aarambam 2lpt binned {outdir + '/LPTconfig'}", shell = True, env = os.environ)

#Finally, here's a helpful executable to combine I/O from individual processes.
sp.run(f"aarambam collate --file_dir {outdir}", shell = True, env = os.environ)
```

### Your own template

Templates are small classes. Write one permutation term of the shape function
$S = B\,(k_1k_2k_3)^2$, with its permutation weight, in the rescaled wavenumbers
$\tilde k = k^{(4-n_s)/3}$; the base class adds the tilt and normalises to $S(1,1,1) = 1$:

```python
class MyTemplate(Am.models.ShapeTemplate):
    def raw_shape(self, k1, k2, k3):
        c = self.args['c']                        #any parameter passed to Decomposer.go(...)
        return k1**2 / (k2 * k3) * 3/6 * (1 + c * k3 / (k1 + k2 + k3))

coeff = Unit.go(MyTemplate, Am.binned.BasicBinnedDecompose, c = 0.5, outdir = outdir)
```

See the [conventions page](https://aarambam.readthedocs.io/en/latest/conventions.html) for the
permutation weights and the amplitude convention ($B_\Phi = 6 f_{\rm NL} B_{\rm template}$).

## Code layout and tests

* `Aarambam/`: the Python package. `models` (templates), `basis/legendre` and `basis/binned`
  (decompositions), `config` (2LPT config files), `postprocess`, `decomposer`, and `cli` (the
  `aarambam` command). `Aarambam.utils` re-exports the config, postprocess and decomposer functions.
* `src/lpt/`: the C sources of the two 2LPT executables, `2LPTBasis` (Legendre/modal, including the
  oscillating-$P(k)$ "ResBasis" mode) and `2LPTBinned`. `common/` is shared by both.
* `tests/`: `python -m pytest tests` runs the Python tests. Tests marked `mpi` run 32^3 ICs and need the
  executables, e.g. `AARAMBAM_LPT_BIN=/path/to/site-packages/Aarambam/bin python -m pytest tests -m mpi`.

Particle IDs are 32-bit by default (up to $2^{31}-1$ particles). Set `export Aarambam_64BIT_IDS=ON` before
`pip install` for larger runs.

## Attribution

If you use the legendre-based decomposition code or derivatives of it, please cite both of [Anbajagane & Lee 2025a](https://arxiv.org/abs/2509.02693) and [Anbajagane & Lee 2025b](https://arxiv.org/abs/2509.02695) who introduced and validated the code for a variety of models. If you use the binned bispectrum method, please cite [Anbajagane & Dalal 2026](https://arxiv.org/abs/2607.24939)

```bibtex
@ARTICLE{Anbajagane:2025:NonlinearI,
       author = {{Anbajagane}, Dhayaa and {Lee}, Hayden},
        title = "{Primordial Physics in the Nonlinear Universe: mapping cosmological collider models to weak-lensing observables}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
         year = 2025,
archivePrefix = {arXiv},
       eprint = {2509.02693},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250902693A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Anbajagane:2025:NonlinearII,
       author = {{Anbajagane}, Dhayaa and {Lee}, Hayden},
        title = "{Primordial Physics in the Nonlinear Universe: signatures of inflationary resonances, excitations, and scale dependence}",
      journal = {arXiv e-prints},
         year = 2025,
archivePrefix = {arXiv},
       eprint = {2509.02695},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv250902695A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{Anbajagane:2026:NonlinearIII,
       author = {{Anbajagane}, Dhayaa and {Dalal}, Neal},
        title = "{Primordial Physics in the Nonlinear Universe: Revealing the oscillating halo bias from cosmological collider models}",
      journal = {arXiv e-prints},
         year = 2026,
archivePrefix = {arXiv},
       eprint = {2607.24939},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260724939A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contact

Please contact Dhayaa Anbajagane (dhayaacosmo at gmail dot com) for any questions on the pipeline.
