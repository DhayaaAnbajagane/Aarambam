CLI
===

The Aarambam package provides a command-line interface (CLI) for generating
and processing non-Gaussian initial conditions. These commands wrap the
underlying Python functions and the compiled 2LPT executables so you can run
pipelines directly from the shell.

.. code-block:: bash

   Aarambam-<command> [options]


Available Commands
------------------

There are a number of available commands. The main ones are for running the built-in
LPT executables. The rest are convenience functions to give example configs,
convert inputs, or clean up output files.

Aarambam-2LPT-Binned
~~~~~~~~~~~~~~~~~~~~

The 2LPT executable for the binned bispectrum decomposition
(``Am.binned.BasicBinnedDecompose``). Use it as

.. code-block:: bash

   Aarambam-2LPT-Binned <config>


or for mpi-runs you can do

.. code-block:: bash

   mpirun -np <N> Aarambam-2LPT-Binned <config>

An example config can be generated as ``Aarambam-make-example-config-binned --config_path <path>``


Aarambam-2LPT-Basis
~~~~~~~~~~~~~~~~~~~

The 2LPT executable for the Legendre (modal) decomposition
(``Am.legendre.BasicLegendreDecompose``). Use it as

.. code-block:: bash

   Aarambam-2LPT-Basis <config>


or for mpi-runs you can do

.. code-block:: bash

   mpirun -np <N> Aarambam-2LPT-Basis <config>

An example config can be generated as ``Aarambam-make-example-config-basis --config_path <path>``

Aarambam-2LPT-ResBasis
~~~~~~~~~~~~~~~~~~~~~~

Similar to ``Aarambam-2LPT-Basis`` but now for the model where the power spectrum can
exhibit oscillations as well.

.. code-block:: bash

   mpirun -np <N> Aarambam-2LPT-ResBasis <config>

An example config can be generated as ``Aarambam-make-example-config-resbasis --config_path <path>``

.. note::

   The config entry ``NumFilesWrittenInParallel`` must not exceed the number of
   MPI ranks ``<N>``; the executables stop with an error otherwise.


Aarambam-make-example-config-binned / -basis / -resbasis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write a complete example parameter file for the corresponding 2LPT executable,
built from the defaults shipped in ``Aarambam/defaults`` (transfer function, glass
file and coefficient tables).

.. code-block:: bash

   Aarambam-make-example-config-binned   --config_path <path>
   Aarambam-make-example-config-basis    --config_path <path>
   Aarambam-make-example-config-resbasis --config_path <path>

``--config_path`` (required) is the file the config is written to. The configs are
the output of ``Am.utils.make_example_config_*()``; edit the paths and parameters
as needed before running.


Aarambam-camb2input
~~~~~~~~~~~~~~~~~~~

Convert a CAMB transfer-function file into the two-column (``k``, ``T(k)``) ASCII
table read by the 2LPT executables (columns 0 and 6 of the CAMB file).

.. code-block:: bash

   Aarambam-camb2input --transfer_path <file>

For ``--transfer_path TransferFunction.dat`` the output is written next to the
input as ``TransferFunctionConverted.dat``.


Aarambam-collate-potential
~~~~~~~~~~~~~~~~~~~~~~~~~~

When a run sets ``SavePotentialField 1``, each MPI rank writes its slab of the
Gaussian and non-Gaussian potentials to separate files in ``OutputDir``. This
command stitches those slabs into single ``Gauss_potential.npy`` and
``Nongauss_potential.npy`` arrays in the same directory and removes the per-rank
files.

.. code-block:: bash

   Aarambam-collate-potential --file_dir <path>
