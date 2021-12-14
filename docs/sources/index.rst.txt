.. Solid-state synthesis science analyzer (S4) documentation master file, created by
   sphinx-quickstart on Fri Dec  3 10:01:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Solid-state synthesis science analyzer (S4)
=======================================================================

This package is designed to solve the modeling of solid-state synthesis
in the synthesis text-mining project. It has the following objectives:

1. Compute thermodynamic quantities for arbitrary compounds by interpolation using DFT data (from the Materials Project, MP).
2. Decompose solid-state reactions into pairwise intermediate reactions by optimizing grand potential.
3. Calculate synthesis features for machine-learning the prediction of solid-state synthesis conditions.
4. Train machine-learning models by properly performing feature engineering, feature selection, and model validation methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   thermodynamics
   cascade
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
========

If you find this package useful, please consider citing the following paper:

* Haoyan Huo, et. al. Machine-learning rationalization and prediction of solid-state
  synthesis conditions, 2021, in preparation.