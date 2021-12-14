.. Solid-state synthesis science analyzer (S4) documentation master file, created by
   sphinx-quickstart on Fri Dec  3 10:01:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Solid-state synthesis science analyzer (S4)
=======================================================================

This package is designed to solve the modeling of solid-state synthesis
in the synthesis text-mining project. It has the following objectives:

1. We provided an universal interface to compute **thermodynamic quantities**
   for **arbitrary compounds** by **interpolation using Materials Project
   entries**.
2. We developed a **reaction driving force cascade model** by decomposing
   solid-state reactions into **pairwise intermediate reactions**.
3. We created four types of synthesis features for **machine-learning prediction
   of solid-state synthesis conditions**.
4. We implemented machine-learning models with proper **feature engineering**,
   **feature selection**, and **model validation** methods.

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