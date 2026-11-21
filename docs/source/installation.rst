Installation
============

There is currently no support for PyPI or conda, so you (unfortunately) need to install from source.
The installation has an extra step (setting environment variables) because of the C libraries that need
to be compiled.

It is recommended to install the code from source via pip install. This can be done as

.. code-block:: bash

   pip install git+https://github.com/DhayaaAnbajagane/Aarambam.git

or you can clone the repo yourself first

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

We recommend that you always point the install at your ``FFTW3`` build by setting the following
environment variable PRIOR to installation. It is the ``--prefix`` used when building FFTW3, i.e.
the directory that contains ``include/fftw3-mpi.h`` and ``lib/``:

.. code-block:: bash

   export Aarambam_FFTW3_ROOT=/path/to/fftw-3.3.10/build

If it is not set, the install looks for ``fftw3-mpi.h`` under ``FFTW3_ROOT``, ``FFTW_DIR``,
``FFTW_ROOT`` and the system paths. (Version 1.0.0 used FFTW2 and ``Aarambam_FFTW2_ROOT``; that
variable is no longer used.)

The install will do its best to find the C compiler, GSL and MPI. If it does not find the right
ones, you can explicitly specify paths with the variables below. On most clusters you will need at
least the GSL and MPI ones (otherwise the MPI library is searched for on ``LD_LIBRARY_PATH`` /
``LIBRARY_PATH`` and the system paths, and GSL only in ``GSL_DIR`` / ``GSL_ROOT`` and the system
paths).

.. code-block:: bash

   export Aarambam_CC=/path/to/mpicc               # default: the first mpicc on PATH
   export Aarambam_GSL_ROOT=/path/to/gsl           # contains include/gsl/ and lib/ (or lib64/)
   export Aarambam_OPENMPI_ROOT=/path/to/openmpi   # contains lib/libmpi.so

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

The C sources of the 2LPT executables live in ``src/lpt`` (``common/`` shared by both methods,
``modal/`` and ``binned/``). They are compiled into ``Aarambam/bin/2LPTBasis`` and
``Aarambam/bin/2LPTBinned`` and are not shipped in the wheel. ``Aarambam/bin/2LPTFFTCheck``
(``src/lpt/check/``) checks the ``band`` FFT backend against FFTW3-MPI on the ranks it is started
on, e.g. ``mpirun -np 2 2LPTFFTCheck 48 32`` (grid size, ``Nsample``); the tests run it.

Running the tests
-----------------

.. code-block:: bash

   python -m pytest tests                                                   # tests marked mpi skip without the executables
   AARAMBAM_LPT_BIN=/path/to/Aarambam/bin python -m pytest tests -m mpi     # 32^3 runs, ~1 min

``AARAMBAM_MPIRUN=/path/to/mpirun`` selects the ``mpirun`` of the MPI the executables were built
with, if the first one on ``PATH`` is another. ``tests/large/run_large_tests.py`` runs larger
rank-independence checks and refuses to start on a busy node.

Installing FFTW3
----------------

Most computing clusters provide FFTW3, but often without its MPI library (``libfftw3_mpi``), or
built against a different MPI than the one you use. In either case (or if there is no FFTW3 at all)
you can install it via the following steps:

.. code-block:: bash

   wget https://www.fftw.org/fftw-3.3.10.tar.gz
   tar -xvzf fftw-3.3.10.tar.gz
   cd fftw-3.3.10; mkdir build;
   ./configure --prefix=${PWD}/build --enable-mpi --enable-shared --enable-sse2 --enable-avx --enable-avx2 MPICC={ADD_PATH_TO_MPICC_EXECUTABLE}
   make -j 8
   make install

You may be able to skip the ``MPICC=`` addition if your computing cluster already supplied
``mpicc`` in an easily findable way, but it must be the same ``mpicc`` (i.e. the same MPI) you
build Aarambam with. The flags:

* ``--enable-mpi`` is required: it builds the MPI-parallel FFTs (``fftw3-mpi.h``,
  ``libfftw3_mpi``), which is what Aarambam includes and links against.
* Keep FFTW's default double precision in this build, i.e. do NOT add ``--enable-float`` or
  ``--enable-long-double``.
* ``--enable-shared`` builds shared libraries; the installed Aarambam executables find them without
  ``LD_LIBRARY_PATH``.
* ``--enable-sse2 --enable-avx --enable-avx2`` build FFTW's SIMD kernels, which make the FFTs
  faster. FFTW checks at run time which of them the CPU can use. ``--enable-avx512`` can be added
  on CPUs that support it, but it is not always a gain: on a Xeon Gold 6248R (which lowers its clock
  for AVX-512 code) it made the single-precision build's runs 1-2% slower and the double one no faster.

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

The ``make -j 8`` step takes a few minutes. The first run with a given grid size and number of ranks
measures the FFT plans once and keeps them (the ``FFTWWisdomDir`` tag, see :doc:`conventions`), so
later runs start straight away.

If you have trouble installing this, please contact Dhayaa Anbajagane (dhayaacosmo at gmail dot com).
I've tested installation at multiple computing clusters now, but there will always be edge-cases :)
