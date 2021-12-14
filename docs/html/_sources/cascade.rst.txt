Reaction cascade model
=======================================================================

All the solid-state reactions are decomposed into pairwise reactions
using the principles developed in our previous works [BianchiniNM2020]_ and [MiuraAM2020]_.
The synthesis reactions are assumed to be performed under a gas reservoir
(:math:`O2`, :math:`CO2`, :math:`N2`, :math:`NH3`) by which gas partial
pressures are kept constant. Therefore, the relevant grand potential change
are calculated as the reaction driving force :math:`\Delta \Phi_{rxn} = \Phi_{products} - \Phi_{reactants}`.
The chemical potential of gas species is calculated using the relation
:math:`\mu (T) = \Delta G_f (T) + RT\ln p`, where :math:`\Delta G_f(T)` is the
experimentally determined Gibbs free energy of formation, :math:`R` is
the gas constant, and :math:`p` is the effective partial pressure of the
gas in normal atmospheric conditions.

The following figure demonstrates an example of the pairwise reaction
cascade construction. Starting with precursor compounds A, B, and C, we
first enumerate all possible pairs of precursor compounds, e.g., :math:`\{\ce{A}, \ce{B}\}`,
:math:`\{\ce{A}, \ce{C}\}`, and :math:`\{\ce{B}, \ce{C}\}`. For each pair
of precursors, we identify all possible pairwise reactions (e.g.,
:math:`\ce{A} + \ce{C} \to \ce{AC}`) by enumerating all MP entries.
These pairwise reactions are normalized per mole of non-gas elements
(e.g., the target compound :math:`SrTiO_3` is normalized by 2). We select
the reaction with lowest reaction driving force :math:`\Delta \Phi_{rxn}`
and use it to consume as much precursor compounds as possible. Note that
the reaction driving force is not necessarily negative to tolerate some
uncertainties in our calculated thermodynamics. This process is repeated
until no possible pairwise reaction could be constructed.

.. figure:: cascade-model.png
   :align: center
   :alt: Cascade model demonstration

   Schematics of the cascade model. We start with the precursors, and each
   time choose the reaction with lowest energies (normalized by metal cations,
   or grand potential). The process is repeated until there is no more possible
   pairwise reactions.

.. [BianchiniNM2020] Bianchini, Matteo, et al. "The interplay between
    thermodynamics and kinetics in the solid-state synthesis of layered
    oxides." Nature materials 19.10 (2020): 1088-1095.
.. [MiuraAM2020] Miura, Akira, et al. "Observing and Modeling the Sequential
    Pairwise Reactions that Drive Solid‐State Ceramic Synthesis." Advanced
    Materials (2021): 2100312.
