CLI
===

Aarambam installs one command, ``aarambam``, which runs the compiled 2LPT
executables and a few helper tools:

.. code-block:: bash

   aarambam <command> [options]

The older ``Aarambam-*`` script names are kept as aliases (see the table at the
end), so existing job scripts keep working.


Running the 2LPT executables
----------------------------

.. code-block:: bash

   aarambam 2lpt binned   <config>
   aarambam 2lpt basis    <config>
   aarambam 2lpt resbasis <config>

or, for MPI runs,

.. code-block:: bash

   mpirun -np <N> aarambam 2lpt binned <config>

* ``binned`` runs ``2LPTBinned``, for tables written by the binned decomposition
  (``Am.binned.BasicBinnedDecompose``).
* ``basis`` runs ``2LPTBasis``, for the Legendre (modal) decomposition
  (``Am.legendre.BasicLegendreDecompose``).
* ``resbasis`` runs the same ``2LPTBasis`` executable. A ResBasis config
  (``Am.utils.make_config_resbasis``) additionally sets ``WhichSpectrum`` to 1
  (linear-:math:`k`) or 2 (log-:math:`k`) and the oscillation parameters
  ``A_res``, ``w_res`` and ``phi_res`` of the primordial power spectrum. With
  ``WhichSpectrum 0`` the result is identical to ``basis``.

A config file has one tag and its value per line; lines starting with ``%`` or
``#``, and anything after the value, are ignored. Values cannot contain spaces
(the config writers refuse such paths). The executables stop with a clear
message, and a non-zero exit code, on a bad or missing config entry, an
over-long path, or a kernel table that does not match the requested filter.
See :doc:`conventions` for size limits.


Example configs
---------------

.. code-block:: bash

   aarambam example-config binned   --config_path <path>
   aarambam example-config basis    --config_path <path>
   aarambam example-config resbasis --config_path <path>

These write the output of ``Am.utils.make_example_config_*()``, built from the
defaults shipped in ``Aarambam/defaults`` (transfer function, glass file and
coefficient tables). Edit the paths and parameters before running. For a binned
decomposition you wrote yourself, ``Am.utils.binned_kernel_inputs(outdir)``
returns the table paths and the matching filter codes to pass to
``make_config_binned``.


camb2input
----------

Convert a CAMB transfer-function file into the two-column (``k``, ``T(k)``) ASCII
table read by the 2LPT executables (columns 0 and 6 of the CAMB file).

.. code-block:: bash

   aarambam camb2input --transfer_path <file>

For ``--transfer_path TransferFunction.dat`` the output is written next to the
input as ``TransferFunctionConverted.dat``.


collate
-------

When a run sets ``SavePotentialField 1``, the Gaussian and non-Gaussian potentials
are written to ``OutputDir``: one file per field on a single rank, or one file per
slab on several ranks. This command stitches them into ``Gauss_potential.npy``
and ``Nongauss_potential.npy`` in the same directory and then removes the
per-rank files.

.. code-block:: bash

   aarambam collate --file_dir <path> [--seed <seed>]

``--seed`` is needed only if the directory holds potential files of several runs.
From Python, ``Am.utils.collate_potential(path)`` returns the arrays instead.


Aliases
-------

==========================================  =========================================
Old name                                    Equivalent
==========================================  =========================================
``Aarambam-2LPT-Binned <config>``           ``aarambam 2lpt binned <config>``
``Aarambam-2LPT-Basis <config>``            ``aarambam 2lpt basis <config>``
``Aarambam-2LPT-ResBasis <config>``         ``aarambam 2lpt resbasis <config>``
``Aarambam-make-example-config-<m>``        ``aarambam example-config <m>``
``Aarambam-camb2input``                     ``aarambam camb2input``
``Aarambam-collate-potential``              ``aarambam collate``
==========================================  =========================================

``aarambam version`` prints the installed version.
