.. raw:: html

   <div style="margin-top: 120px"></div>

.. image:: LOGO_light.png
   :class: only-light
   :width: 85%
   :align: center

.. image:: LOGO_dark.png
   :class: only-dark
   :width: 85%
   :align: center

.. raw:: html

   <div style="margin-top: 50px"></div>


**Aarambam** is a tool for generating non-Gaussian initial conditions for N-body simulations from arbitrary
bispectrum templates. It decomposes a template into separable pieces and feeds them to a modified 2LPT
initial-conditions code. Two decompositions are provided:

* the Legendre (modal) decomposition, introduced and validated in
  `Anbajagane & Lee 2025a <https://arxiv.org/abs/2509.02693>`_ and
  `Anbajagane & Lee 2025b <https://arxiv.org/abs/2509.02695>`_, based on modifications to existing
  pipelines for bispectrum modal decomposition (CMB-BEST) and for 2LPT initial conditions (2LPTic);
* the binned decomposition, introduced in `Anbajagane & Dalal 2026 <https://arxiv.org/abs/2607.24939>`_.

Please cite the corresponding papers if you use the code (BibTeX entries are in the
`README <https://github.com/DhayaaAnbajagane/Aarambam>`_).

.. raw:: html

   <div style="margin-top: 50px"></div>

**Contents**

.. toctree::
   :maxdepth: 2

   installation
   cli
   conventions
   performance
   api_reference


.. raw:: html

   <div style="margin-top: 50px"></div>

**Indices and tables**

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`