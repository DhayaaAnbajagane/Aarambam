Installation
============

There is currently no support for PyPi or conda, so you (unfortunately) need to install from source.
The installation has two extra steps (setting env variables) due to the C libraries than need to be compiled.
I copy below the readme installation instructions.


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

There are a number of dependencies, particularly due to the 2LPT (2nd order Lagrangian Pertubration Theory) method provided in this package, namely `MPI`, `FFTW2` (NOT `FFTW3`. Using `FFTW2` a legacy dependence from `2LPTIC` days), and `GSL`. The install will generally do its best to find the right compiled objects for these dependencies. Please set the following environment variables PRIOR to installation

```bash
export Aarambam_FFTW2_ROOT=#path to where your /build directory is
```

If the install is not finding your correct installs of C-compilers, GSL etc. then you can explicitly specify paths with

```bash
export Aarambam_CC=
export Aarambam_GSL_ROOT=
export Aarambam_OPENMPI_ROOT=
```

If your computing cluster does not provide an install of FFTW2 (most do not since it is old...) then you can install it via the following steps:

```bash
wget https://www.fftw.org/fftw-2.1.5.tar.gz
tar -xvzf fftw-2.1.5.tar.gz
cd fftw-2.1.5; mkdir build;
./configure --prefix=${PWD}/build --enable-threads --enable-openmp --enable-mpi --enable-type-prefix CC={ADD_PATH_TO_MPICC_EXECUTABLE}
make
make install
```
You may be able to skip the `CC=` addition if your computing cluster already supplied `mpicc` in an easily findable way. The above `make` command will take a while to run (FFTW2 has a lot of libraries to install). FFTW2 is a hard requirement as this piece of LPT code originates from before the 2000s, and FFTW2 was officially deprecated only in 1999.

If you have trouble installing this, please contact Dhayaa (details below). I've tested installation at multiple computing clusters now, but there will always be edge-cases:)