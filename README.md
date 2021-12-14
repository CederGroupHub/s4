## Solid-state synthesis science analyzer (S4)

This package is designed to model solid-state syntheses in the [synthesis text-mining project](https://ceder.berkeley.edu/text-mined-synthesis/). 
It has the following objectives:

1. Compute thermodynamic quantities for arbitrary compounds by interpolation using DFT data (from the Materials Project, MP).
2. Decompose solid-state reactions into pairwise intermediate reactions by optimizing grand potential.
3. Calculate synthesis features for machine-learning the prediction of solid-state synthesis conditions.
4. Train machine-learning models by properly performing feature engineering, feature selection, and model validation methods.

## Thermodynamic quantity calculation

In this package, the thermodynamic quantity can be calculated using either experimental databases and DFT data from MP.
Because the lack of experimental data, not every compound's experimental thermodynamic data can be found. 

On the other hand, DFT databases contain enough entries covering much of the chemical space. We also perform 
interpolation if the compound has no direct match in the DFT database. Therefore, thermodynamic quantities of most
compounds can be calculated in this way.

### Using experimental databases

We used the FREED database to compute thermodynamic quantities using experimentally determined data. 
[FREED database](https://www.thermart.net/freed-thermodynamic-database/) is an electronic compilation of the U.S. Bureau 
of Mines (USBM) experimental thermodynamic data. We don't perform any interpolation to ensure the experimental 
thermodynamic data are accurate.

As an example, the following code computes the Gibbs formation energy and formation enthalpy for BaCO3.

```python
from s4.thermo.exp.freed import database

print(database.dhf('BaCO3', 300, unit='ev/atom'))
# Prints -2.509443985939046

print(database.dgf('BaCO3', 300, unit='ev/atom'))
# Prints -2.345862145732135
```

### Using DFT data in MP

Since not every compound has direct match in MP, we perform interpolation of compounds, which is developed by 
[Chris Bartel](https://cjbartel.github.io/). Please read the [documentation](insert link here) for the details of this
interpolation algorithm.

As an example, the following code computes the Gibbs formation energy and formation enthalpy for BaCO3.

```python
from s4.thermo.calc.mp import database

print(database.dhf('BaCO3', 300, unit='ev/atom'))
# Prints -2.5211004719999996
print(database.dgf('BaCO3', 300, unit='ev/atom'))
# Prints -2.361179038215705
```

For BaCO3, the experimental values and the DFT-derived values are very close. This is because we performed additional 
corrections to the thermodynamic data, please see the [documentation](insert link here) for details.

## Thermodynamic pairwise reaction cascade construction

The basic assumption of this cascade construction is the maximum reaction driving force hypothesis, which states that 
the pairwise reaction happening on reactant interfaces are the ones with the maximum reaction driving force (grand 
potential). This is demonstrated in the paper by [Bianchini et al.](https://www.nature.com/articles/s41563-020-0688-6.pdf)
and [Miura et al.](https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202100312). The details of this algorithm can 
be found in the [documentation](insert link here).

The following demonstrates an example from [Bianchini et al.](https://www.nature.com/articles/s41563-020-0688-6.pdf) on 
the phase evolution of Na2O2 + CoO = NaxCoO2.

```python
from s4.tmr import ReactionEnergies, MaterialWithEnergy, MPInterpolatedMaterial
from s4.cascade.analysis import compute_cascade
from s4.thermo.calc.mp import query_system
from pymatgen import Composition as C

reaction = ReactionEnergies(
    target=C('Na2(CoO2)3'),
    vars_sub={},
    species=[
        MaterialWithEnergy(
            thermo=MPInterpolatedMaterial(
                compositions=[C('Na2(CoO2)3')], amounts=[1./3], mp_entries=[query_system('Na2(CoO2)3')[0]]),
            composition=C('Na2(CoO2)3'), is_target=True, side='product', amt=1./3),
        MaterialWithEnergy(
            thermo=MPInterpolatedMaterial(
                compositions=[C('CoO')], amounts=[1.], mp_entries=[query_system('CoO')[0]]),
            composition=C('CoO'), is_target=False, side='reactant', amt=1.),
        MaterialWithEnergy(
            thermo=MPInterpolatedMaterial(
                compositions=[C('Na2O2')], amounts=[1./3], mp_entries=[query_system('Na2O2')[0]]),
            composition=C('Na2O2'), is_target=False, side='reactant', amt=1./3),
    ]
)

compute_cascade(reaction, [500]*10, only_icsd=False)

# Prints
# [{'driving_force': -0.4195086425659669,
#   'temperature': 500,
#   'previous_vessel': {1.0 CoO, 0.3333 Na2O2},
#   'current_vessel': {0.3333 CoO, 0.6667 Na1Co1O2},
#   'reason': 'cascade: determined by minimizing dG/m.a'},
#  {'driving_force': -0.037765379636589554,
#   'temperature': 500,
#   'previous_vessel': {0.3333 CoO, 0.6667 Na1Co1O2},
#   'current_vessel': {0.5 Na1Co1O2, 0.1667 Na1Co3O6},
#   'reason': 'cascade: determined by minimizing dG/m.a'},
#  {'driving_force': -0.021780617177087437,
#   'temperature': 500,
#   'previous_vessel': {0.5 Na1Co1O2, 0.1667 Na1Co3O6},
#   'current_vessel': {0.2222 Na1Co1O2, 0.1111 Na4Co7O14},
#   'reason': 'cascade: determined by minimizing dG/m.a'},
#  {'driving_force': -0.0015646056138710655,
#   'temperature': 500,
#   'previous_vessel': {0.2222 Na1Co1O2, 0.1111 Na4Co7O14},
#   'current_vessel': {0.0667 Na4Co7O14, 0.1333 Na3Co4O8},
#   'reason': 'cascade: determined by minimizing dG/m.a'},
#  {'driving_force': 0.0018870757762980397,
#   'temperature': 500,
#   'previous_vessel': {0.0667 Na4Co7O14, 0.1333 Na3Co4O8},
#   'current_vessel': {0.1111 Na3Co4O8, 0.1111 Na3Co5O10},
#   'reason': 'cascade: determined by minimizing dG/m.a'},
#  {'driving_force': 0.08069345991540304,
#   'temperature': 500,
#   'previous_vessel': {0.1111 Na3Co4O8, 0.1111 Na3Co5O10},
#   'current_vessel': {0.3333 Na2Co3O6},
#   'reason': 'cascade: determined by minimizing dG/m.a'}]
```

## Computing synthesis features for solid-state synthesis reactions

In this package, we compute four types of synthesis features (133 features in total). 

**Precursor compound properties**. The first type of features (12 in total) are the average/ minimum/ maximum/ difference 
of melting points, standard enthalpy of formation, standard Gibbs free energy of formation of precursors. The melting 
points are retrieved from the NIST Chemistry WebBook + Wikipedia, while the experimental thermodynamic properties were 
retrieved from the FREED database. 

**Target compound compositional features**. The second type of features are 74 indicator variables representing the 
presence of different chemical elements in target compounds. 

**Reaction thermodynamics features**. We used 32 thermodynamic features, including the total reaction driving force, 
first and last pairwise reaction driving force, and the ratio between first/last pairwise reaction driving force and the 
total reaction driving force, evaluated at different temperatures T=800, 900, 1000, 1100, 1200, and 1300 degrees Celsius. 
We also calculated the slope of total/first driving forces by assuming they are linear with respect to temperature 
and used the slopes as additional features.

**Experiment-adjacent features**. The fourth type of features are 15 experiment-adjacent features, i.e., indicator 
variables representing whether certain devices, experiment procedures, and aiding materials were used in the synthesis.

To understand how they are calculated, please refer to [features.py](s4/ml/features.py).
The following code calculates the features for the reaction above as a dictionary.

```python
from s4.ml.features import Featurizer

featurizer = Featurizer()
features = featurizer.featurize(reaction, exp_t=800, exp_time=6)
```

## Citation

If you find this package useful, please consider citing the following paper:

* Haoyan Huo, et. al. Machine-learning rationalization and prediction of solid-state synthesis
  conditions, 2021, in preparation.