"""
This script computes the fitted correction for CO3 anion.

The correction is -1.2485 ev/CO3 for fitting directly the
formation enthalpy of carbonates.

The correction is -1.2769 ev/CO3 for fitting directly the
difference between formation enthalpy of carbonates and
oxides.
"""
import matplotlib.pyplot as plt
import numpy
import pandas
from pymatgen import Composition as C
from scipy import optimize

from s4.thermo.calc.mp import rester
from s4.thermo.exp.freed import database

__author__ = 'Haoyan Huo'
__email__ = 'haoyan.huo@lbl.gov'
__maintainer__ = 'Haoyan Huo'

C_ENERGY = -9.2268
O_ENERGY = -4.9480


def prepare_data_to_fit():
    """Generate data collected from MP to fit corrections."""
    carbonates = [
        # Group 1
        'Li2CO3', 'Na2CO3', 'K2CO3', 'Cs2CO3',
        # Group 2
        'MgCO3', 'CaCO3', 'SrCO3', 'BaCO3',  # CaCO3 2 polymorphs: arag/calcite BeCO3

        # Group VIIB
        'MnCO3', 'FeCO3',
        # Group IB
        'Ag2CO3',
        # Group IIB
        'ZnCO3', 'CdCO3',
        # Group IIIA
        'Tl2CO3',
        # Group IVA
        'PbCO3',
    ]

    results = []

    for comp in carbonates:
        print('Querying data for', comp)

        entry = rester.get_entries(comp, sort_by_e_above_hull=True)[0]
        entry.normalize()
        natoms = sum(entry.composition.values())

        metal_s = next(x.symbol for x in entry.composition if x.symbol not in {'O', 'C'})
        metal = rester.get_entries(metal_s, sort_by_e_above_hull=True)[0]
        metal.normalize()
        metal_natoms = sum(metal.composition.values())

        oxide_s = C(
            {metal_s: entry.composition[metal_s], 'O': entry.composition['O'] - 2}).reduced_formula
        oxide = rester.get_entries(oxide_s, sort_by_e_above_hull=True)[0]
        oxide.normalize()
        oxide_natoms = sum(oxide.composition.values())

        data = {
            'composition_carbonate': comp,
            'composition_metal': metal_s,
            'composition_oxide': oxide_s,

            'carbonate_mp_id': entry.entry_id,
            'carbonate_mp_E (ev/atom)': entry.energy / natoms,
            'carbonate_mp_E_uncorrected (ev/atom)': entry._energy / natoms,  # pylint: disable=protected-access
            'carbonate_mp_corrections': [[x.name, x.value] for x in entry.energy_adjustments],

            'metal_mp_id': metal.entry_id,
            'metal_mp_E (ev/atom)': metal.energy / metal_natoms,
            'metal_mp_E_uncorrected (ev/atom)': metal._energy / metal_natoms,  # pylint: disable=protected-access
            'metal_mp_corrections': [[x.name, x.value] for x in metal.energy_adjustments],

            'oxide_mp_id': oxide.entry_id,
            'oxide_mp_E (ev/atom)': oxide.energy / oxide_natoms,
            'oxide_mp_E_uncorrected (ev/atom)': oxide._energy / oxide_natoms,  # pylint: disable=protected-access
            'oxide_mp_corrections': [[x.name, x.value] for x in oxide.energy_adjustments],

            'carbonate_freed_dhf': database.dhf(comp, temperature=298.15, unit='ev/atom'),
            'oxide_freed_dhf': database.dhf(oxide_s, temperature=298.15, unit='ev/atom'),
        }
        results.append(data)
    table = pandas.DataFrame(results)

    return table


def ufunc(data_x, bias):
    """Intercept only model."""
    return data_x + bias


def fit_only_bias(data_x, data_y, labels):
    """Fit a intercept only model."""
    popt = optimize.curve_fit(ufunc, data_x, data_y)[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(data_x, data_y, marker='x')
    _min, _max = min(data_x), max(data_x)
    plt.plot([_min, _max], [_min + popt, _max + popt], label='Bias: %.3f' % (popt,))
    for i in range(len(labels)):
        plt.text(data_x[i], data_y[i], labels[i], fontsize=16)
    plt.legend(loc='upper left', fontsize=16)

    print('Fit bias is %.4f ev/CO3' % popt)
    return popt


def fit_carbonate_direct(table):
    natoms = table['composition_carbonate'].apply(lambda c: sum(C(c).values()))
    data_y = table['carbonate_freed_dhf'] * natoms
    data_x = (
            table['carbonate_mp_E (ev/atom)'] * natoms
            - -0.70229 * 3
            - (C_ENERGY + O_ENERGY * 3)
            - table['metal_mp_E (ev/atom)'] * (natoms - 4)
    )

    fix_co3 = fit_only_bias(data_x.values, data_y.values, table.composition_carbonate.values)
    plt.xlabel('Calculation enthalpy (0K)', fontsize=16)
    plt.ylabel('Experimental enthalpy (298.15K)', fontsize=16)
    plt.title('CO3(-2) fitted correction %.4f' % fix_co3)
    error = (data_y - data_x - fix_co3) / natoms
    print('Error mean: %.3f ev/atom, std: %.3f ev/atom' % (
        numpy.mean(error).item(), numpy.std(error).item()))
    plt.show()


def fit_carbonate_oxides(table):
    """
    For reaction $Ba + C + 1.5 O2 == BaCO3$ and $Ba + O == BaO$
    $HE_{BaCO3} = E_{BaCO3} + Fix_{CO3} - E_{Ba} - E_{C} - 3*E_{O}$
    and,
    $HE_{BaO} = (E_{BaO}+E_{CorrOxide}) - E_{Ba} - 1*E_{O}$

    Thus,
    $HE_{BaCO3} - HE_{BaO} = E_{BaCO3} - (E_{BaO}+E_{CorrOxide}) + Fix_{CO3} - E_{C} - 2*E_{O}$

    Let:
    $x = HE_{BaCO3} - HE_{BaO}$
    $y = E_{BaCO3} - (E_{BaO}+E_{CorrOxide})$

    We have the linear regression of $y$ on to $x$:

    $slope = 1$
    $intercept = E_{C} + 2*E_{O} - Fix_{CO3}$
    """
    natoms = table['composition_carbonate'].apply(lambda c: sum(C(c).values()))
    oxide_natoms = table['composition_oxide'].apply(lambda c: sum(C(c).values()))

    data_y = table['carbonate_freed_dhf'] * natoms - table['oxide_freed_dhf'] * oxide_natoms
    data_x = (
                table['carbonate_mp_E (ev/atom)'] * natoms
                - -0.70229 * 3
                - (C_ENERGY + O_ENERGY * 3)
                - table['metal_mp_E (ev/atom)'] * (natoms - 4)
        ) - (
                table['oxide_mp_E (ev/atom)'] * oxide_natoms
                - O_ENERGY
                - table['metal_mp_E (ev/atom)'] * (oxide_natoms - 1)
        )

    fix_co3 = fit_only_bias(data_x.values, data_y.values, table.composition_carbonate.values)
    plt.xlabel('Calculation enthalpy rel oxide (0K)', fontsize=16)
    plt.ylabel('Experimental enthalpy rel oxide (298.15K)', fontsize=16)
    plt.title('CO3(-2) fitted correction %.4f' % fix_co3)
    error = (data_y - data_x - fix_co3) / natoms
    print('Error mean: %.3f ev/atom, std: %.3f ev/atom' % (
        numpy.mean(error).item(), numpy.std(error).item()))
    plt.show()


def main():
    load_table = True

    if not load_table:
        table = prepare_data_to_fit()
        table.to_csv('000.fit_carbonates.csv')
    else:
        table = pandas.read_csv('000.fit_carbonates.csv')

    # Do not use these ones because they are inaccurate!
    use_fit = table.loc[table.composition_carbonate.apply(
        lambda x: x not in {'FeCO3', 'MnCO3', 'PbCO3', 'Tl2CO3', 'Ag2CO3'})]

    fit_carbonate_direct(use_fit)
    fit_carbonate_oxides(use_fit)


if __name__ == '__main__':
    main()
