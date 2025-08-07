<p>&nbsp;</p>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/LOGO_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/source/LOGO_light.png">
  <img alt="Logo" src="docs/source/LOGO_dark.png" title="Logo">
</picture>
<p>&nbsp;</p>
<p>&nbsp;</p>

[![License](https://img.shields.io/badge/license-GPL-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

`Aarambam` (pronounced "Aah-rum-bum", named after the Tamil word for beginnings) is a pipeline for generating initial conditions (ICs) corresponding to arbitrary bispectrum templates. It is an end-to-end pipeline that can provide the final ICs given an analytic bispectrum template. This is a largely modified version of two packages --- [CMB-BEST](https://github.com/Wuhyun/CMB-BEST/tree/main) by Wuhyun Sohn, and [2LPTPNG](https://github.com/dsjamieson/2LPTPNG/tree/main) by Drew Jamieson (which itself derives from [2LPTIC](https://github.com/manodeep/2LPTic) and previous codebases) --- so please see those packages for the original implementations. This package now provides all routines fo the decomposition of arbitrary bispectra into separable functions, and the subsequent generation of initial conditions corresponding to these functions. 

The method implemented in `Aarambam` was introduced in Anbajagane & Lee (2025a) and Anbajagane & Lee (2025b). Citations to the same are requested if you are using this code for a publication. Contact dhayaa at uchicago dot edu if you have any questions. Happy simulating!


(The first two letters of the logo are from the Tamil script. You can do a process of elimination using the English letters to guess what those letters' sounds are ;) )

## Environment
The python part of the code has a few dependencies --- namely `numpy`, `scipy`, `tqdm`, `joblib`, and `threadpoolctl`. The last one is more non-standard but is very helpful in managing oversubscription. I promise it is easy to install! The C-level code has its own dependencies, I list them below in the Installation instructions.

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

There are a number of dependencies, particularly due to the 2LPT (2nd order Lagrangian Pertubration Theory) method provided in this package, namely `MPI`, `FFTW2` (NOT `FFTW3`, this is a legacy dependence from `2LPTIC` days), and `GSL`. The install will generally do its best to find the right compiled objects for these dependencies. Please set the following environment variables PRIOR to installation

```bash
export Aarambam_FFTW2_ROOT=#path to where your /build directory is
```

If the install is not finding your correct installs of C-compilers, GSL etc. then you can explicitly specify paths with

```bash
export Aarambam_CC=
export Aarambam_GSL_ROOT=
export Aarambam_OPENMPI_ROOT=
```

## Example

We provide examples of how to use this code in [this notebook](examples/BasisDecomposition.ipynb) and [this python script](examples/MakeICs.py). I copy the snippets of the script below.

```python
import Aarambam as Am
import numpy as np, subprocess as sp

#This is the main code needed for decomposing a given template. Swap "ScalarI" with
#other options to decompose more models. It is very easy to implement your own :)
#If you pass "outdir" then this will also write the coefficient tables in the right
#format into those directories
Unit   = Am.utils.Decomposer(N_modes = N_modes, n_s = n_s, Lbox = Lbox, Nmax = Nmax, ModeTol = ModeTol, MaxModeCount = MaxModeCount)
coeffL = Unit.go(Am.models.Local,   Am.basis.BasicBasisDecompose, outdir = outdir)
coeffS = Unit.go(Am.models.ScalarI, Am.basis.BasicBasisDecompose, mass = 1, outdir = outdir)

#Use the helper function to make the config in the right format
#and write it to dir. You will need to pass a bunch of
#values into make_config
CONFIG = Am.utils.make_config(...)
with open(outdir + '/LPTconfig', 'w') as f: f.write(CONFIG)

#Doing a pip install will provide you with executables in your env.
#The Aarambam-2LPT-Basis one is the main LPT one for us. We run it with
#subprocess just for simplicity. You could also run it from cmd line
sp.run(f"mpirun -np {Nprocs} Aarambam-2LPT-Basis {outdir + '/LPTconfig'}", shell = True, env = os.environ)

#Finally, here's a helpful executable to combine I/O from individual processes.
sp.run(f"Aarambam-collate-potential --file_dir {outdir}", shell = True, env = os.environ)
```