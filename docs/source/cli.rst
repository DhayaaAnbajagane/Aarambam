CLI
===

The Aarambam package provides a command-line interface (CLI) for generating
and processing non-Gaussian initial conditions. These commands wrap the
underlying Python functions so you can run pipelines directly from the shell.

Usage
-----

.. code-block:: bash

   Aarambam-<command> [options]

Available Commands
------------------

Aarambam-2LPT-Basis
~~~~~~~~~~~~~~~~~

The main cmdline executable for running the 2LPT code. Use it as

.. code-block:: bash

   Aarambam-2LPT-Basis <config>


or for mpi-runs you can do

.. code-block:: bash

   mpirun -np <N> Aarambam-2LPT-Basis <config>

An example config can be generated as :bash:`Aarambam-make-example-config-basis --config_path <path>`

Aarambam-2LPT-ResBasis
~~~~~~~~~~~~~~~~~

Similar to ``Aarambam-2LPT-Basis`` but now for the model where the power spectrum can
exhibit oscillations as well.

.. code-block:: bash

   Aarambam-2LPT-Basis <config>

An example config can be generated as :bash:`Aarambam-make-example-config-resbasis --config_path <path>`


Aarambam-collate-potential
~~~~~~~~~~~~~~~~~

Collates individual potential files (during mpiruns) into a single npy output.

.. code-block:: bash

   Aarambam-collate-potential --file_dir <path>