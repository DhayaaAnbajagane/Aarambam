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
``MPI``, ``FFTW2`` (NOT ``FFTW3``; ``FFTW2`` is a legacy dependence from the ``2LPTic`` days) and
``GSL``. The Cython extension used by the Legendre decomposition also needs a compiler with
``OpenMP`` support.

``FFTW2`` is never searched for automatically, so you must ALWAYS set the following environment
variable PRIOR to installation. It points to the ``--prefix`` used when building FFTW2, i.e. the
directory that contains ``include/dfftw.h`` and ``lib/``:

.. code-block:: bash

   export Aarambam_FFTW2_ROOT=/path/to/fftw-2.1.5/build

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

The C sources of the 2LPT executables live in ``src/lpt`` (``common/`` shared by both methods,
``modal/`` and ``binned/``). They are compiled into ``Aarambam/bin/2LPTBasis`` and
``Aarambam/bin/2LPTBinned`` and are not shipped in the wheel.

Running the tests
-----------------

.. code-block:: bash

   python -m pytest tests                                                   # tests marked mpi skip without the executables
   AARAMBAM_LPT_BIN=/path/to/Aarambam/bin python -m pytest tests -m mpi     # 32^3 runs, ~10 s

``tests/large/run_large_tests.py`` runs larger rank-independence checks and refuses to start on a
busy node.

Installing FFTW2
----------------

If your computing cluster does not provide an install of FFTW2 (most do not since it is old...)
then you can install it via the following steps:

.. code-block:: bash

   wget https://www.fftw.org/fftw-2.1.5.tar.gz
   tar -xvzf fftw-2.1.5.tar.gz
   cd fftw-2.1.5; mkdir build;
   ./configure --prefix=${PWD}/build --enable-mpi --enable-type-prefix CC={ADD_PATH_TO_MPICC_EXECUTABLE}
   make
   make install

You may be able to skip the ``CC=`` addition if your computing cluster already supplied ``mpicc``
in an easily findable way. Both flags are required:

* ``--enable-mpi`` builds the MPI-parallel FFTs used by the 2LPT code.
* ``--enable-type-prefix`` gives the double-precision libraries and headers their ``d`` prefix
  (``drfftw_mpi.h``, ``libdrfftw_mpi``, ...), which is what Aarambam includes and links against.

The ``make`` command will take a while to run (FFTW2 has a lot of libraries to install). FFTW2 is a
hard requirement because the LPT code descends from ``2LPTic``, which was written against the FFTW2
MPI interface (FFTW 2.1.5, from 2003, is the last FFTW2 release).

If you have trouble installing this, please contact Dhayaa Anbajagane (dhayaacosmo at gmail dot com).
I've tested installation at multiple computing clusters now, but there will always be edge-cases :)
