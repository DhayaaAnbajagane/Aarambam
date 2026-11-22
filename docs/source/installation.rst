Installation
============

Aarambam is not on PyPI or conda yet, so install it from source with pip. Because it compiles C
executables, first set the environment variables of `C dependencies`_ below. Then

.. code-block:: bash

   pip install git+https://github.com/DhayaaAnbajagane/Aarambam.git

or, from a clone,

.. code-block:: bash

   git clone https://github.com/DhayaaAnbajagane/Aarambam.git
   cd Aarambam
   pip install .

``pip`` installs the Python dependencies (``numpy``, ``scipy`` >= 1.12, ``tqdm``, ``joblib`` and
``threadpoolctl``) automatically; ``cython`` is only needed at build time.

C dependencies
--------------

The 2LPT (2nd order Lagrangian Perturbation Theory) executables shipped with this package need
``MPI``, ``FFTW3`` (double precision, built with its MPI library) and ``GSL``. The Cython
extension used by the Legendre decomposition also needs a compiler with ``OpenMP`` support.

Point the install at your ``FFTW3`` build before installing: the ``--prefix`` used when building
FFTW3, i.e. the directory that contains ``include/fftw3-mpi.h`` and ``lib/``:

.. code-block:: bash

   export Aarambam_FFTW3_ROOT=/path/to/fftw-3.3.10/build

If it is not set, the install looks for ``fftw3-mpi.h`` under ``FFTW3_ROOT``, ``FFTW_DIR``,
``FFTW_ROOT``, the include directories of ``mpicc`` and the system paths. (Version 1.0.0 used FFTW2 and ``Aarambam_FFTW2_ROOT``; that
variable is no longer used.)

The install looks for the C compiler, GSL and MPI itself. If it picks the wrong ones, set the paths
below; on most clusters at least the GSL and MPI ones are needed (otherwise MPI is searched for in
the library directories of ``mpicc``, next to it, on ``LD_LIBRARY_PATH`` / ``LIBRARY_PATH`` and the system paths, GSL in ``GSL_DIR`` / ``GSL_ROOT`` and
the system paths).

.. code-block:: bash

   export Aarambam_CC=/path/to/mpicc               # default: the first mpicc on PATH
   export Aarambam_GSL_ROOT=/path/to/gsl           # contains include/gsl/ and lib/ (or lib64/)
   export Aarambam_OPENMPI_ROOT=/path/to/mpi       # Open MPI or MPICH; contains lib/libmpi.so (or libmpich.so)

Snapshots use 32-bit particle IDs by default, which allows at most :math:`2^{31}-1` particles
(the executables stop with a message beyond that). For larger runs, build with 64-bit IDs:

.. code-block:: bash

   export Aarambam_64BIT_IDS=ON    # then pip install as usual

To compile the 2LPT executables for the CPU of the machine you install on (``-march=native``:
AVX2 / AVX-512 and fused multiply-adds; a few per cent faster), set

.. code-block:: bash

   export Aarambam_NATIVE=ON       # only if the nodes you run on have the same CPU type

The executables then stop with "illegal instruction" on CPUs without those instruction sets (for
example AVX-512 code on AMD nodes), so leave it off when login and compute nodes differ.

``export Aarambam_PYTHON_EXTENSION=OFF`` builds only the C executables, without the Cython extension
(for example where no Python development files are installed); the Legendre decomposition then cannot
be used.

The C sources of the 2LPT executables live in ``src/lpt`` (``common/`` shared by both methods,
``modal/`` and ``binned/``). They are compiled into ``Aarambam/bin/2LPTBasis`` and
``Aarambam/bin/2LPTBinned`` and are not shipped in the wheel. ``Aarambam/bin/2LPTFFTCheck``
(``src/lpt/check/``) tests the ``band`` FFT backend (see :doc:`conventions`).

Running the tests
-----------------

.. code-block:: bash

   python -m pytest tests                                                   # tests marked mpi skip without the executables
   AARAMBAM_LPT_BIN=/path/to/Aarambam/bin python -m pytest tests -m mpi     # 32^3 runs, ~1 min

``AARAMBAM_MPIRUN=/path/to/mpirun`` selects the ``mpirun`` of the MPI the executables were built
with, if the first one on ``PATH`` is another. ``AARAMBAM_TEST_INSTALLED=1 python -m pytest tests``
tests the installed package instead of the source tree, including its compiled Legendre extension;
the GitHub workflow does this after ``pip install .``. ``tests/large/run_large_tests.py``
runs larger rank-independence checks and refuses to start on a busy node.

Installing FFTW3
----------------

Build FFTW3 from source, against the MPI you build Aarambam with: that is the supported route, and the
GitHub workflow tests it with the MPI and GSL of Ubuntu (Open MPI and MPICH), Rocky Linux 8 and 9 and
openSUSE Leap 15.6. On a cluster, load its MPI and GSL modules; on your own machine, install them:

.. code-block:: bash

   # Debian, Ubuntu
   sudo apt-get install openmpi-bin libopenmpi-dev libgsl-dev
   # RHEL, Rocky, Alma 8 or 9 (gsl-devel is in PowerTools / CRB); then module load mpi/openmpi-x86_64
   sudo dnf install openmpi-devel gsl-devel
   # openSUSE, SLES; then put /usr/lib64/mpi/gcc/openmpi4/bin on PATH
   sudo zypper install openmpi4-devel gsl-devel

On NERSC Perlmutter (not tested here), build FFTW3 below with ``MPICC=cc`` and set
``export Aarambam_CC=cc`` (the Cray compiler wrapper, which adds Cray MPICH); ``module load gsl``
sets the ``GSL_DIR`` the install looks in.

Then build FFTW3:

.. code-block:: bash

   wget https://www.fftw.org/fftw-3.3.10.tar.gz
   tar -xvzf fftw-3.3.10.tar.gz
   cd fftw-3.3.10; mkdir build;
   ./configure --prefix=${PWD}/build --enable-mpi --enable-shared --enable-sse2 --enable-avx --enable-avx2 MPICC={ADD_PATH_TO_MPICC_EXECUTABLE}
   make -j 8
   make install

``MPICC=`` can be left out if the right ``mpicc`` is first on ``PATH``; it must be the same MPI you
build Aarambam with. The flags:

* ``--enable-mpi`` is required: it builds the MPI-parallel FFTs (``fftw3-mpi.h``,
  ``libfftw3_mpi``), which is what Aarambam includes and links against.
* Keep FFTW's default double precision in this build, i.e. do NOT add ``--enable-float`` or
  ``--enable-long-double``.
* ``--enable-shared`` builds shared libraries; the installed Aarambam executables find them without
  ``LD_LIBRARY_PATH``.
* ``--enable-sse2 --enable-avx --enable-avx2`` build FFTW's SIMD kernels (chosen at run time to
  match the CPU), which make the FFTs faster. ``--enable-avx512`` is not always a gain: on a Xeon
  Gold 6248R (which lowers its clock for AVX-512 code) it made single-precision runs 1-2% slower and
  double-precision ones no faster.

Optionally, build FFTW a second time in single precision, into the same prefix, for the
``NGPrecision float`` option (see :doc:`performance`). From a fresh copy of the source:

.. code-block:: bash

   mkdir float_build; tar -xvzf fftw-3.3.10.tar.gz -C float_build; cd float_build/fftw-3.3.10
   ./configure --prefix={THE_SAME_PREFIX_AS_ABOVE} --enable-float --enable-shared --enable-sse --enable-sse2 --enable-avx --enable-avx2
   make -j 8
   make install

This one needs no ``--enable-mpi`` (the code does its own single-precision distributed FFT).
``--enable-float`` builds ``libfftw3f`` next to ``libfftw3``; the Aarambam install finds it there
and enables the option (without it, ``NGPrecision float`` stops with a message).

Each ``make -j 8`` takes a few minutes.

If you have trouble installing this, please contact Dhayaa Anbajagane (dhayaacosmo at gmail dot com).
I've tested installation at multiple computing clusters now, but there will always be edge-cases :)
