import pickle
from argparse import ArgumentParser
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
import synthesis_dataset
from pymatgen.core import Composition as _C
from tqdm import tqdm

from s4.cascade.balance import quick_balance
from s4.thermo.constants import ATM_GAS_PARTIALS
from s4.tmr.entry import MaterialWithEnergy, ReactionEntry, ReactionEnergies
from s4.tmr.thermo import MPInterpolatedMaterial, GasMaterial

possible_gases = {
    _C('CO2'),
    _C('O2'),
}


def from_pcd_entry(row, k, override_fugacity=None):
    """
    :param row: The row of PCD data
    :param k: Key of entry
    :param override_fugacity: If not None, set the fugacity of certain gases.
    """
    # Prepare materials
    target, formula, substitution, equation = row.reactions
    precursors = row.precursors
    open_comp = list(possible_gases)
    fugacity = ATM_GAS_PARTIALS.copy()
    fugacity.update({_C(c): v for c, v in (override_fugacity or {}).items()})

    # Try to balance
    precursor_amts, open_comp_amts = quick_balance(precursors, open_comp, target)

    # Target compound
    target_thermo = MPInterpolatedMaterial.from_material_dict({'thermo': [{'formula': target}]})
    species = [MaterialWithEnergy(
        thermo=target_thermo, composition=target_thermo.composition,
        is_target=True, side='product', amt=1)]

    # Precursor compounds
    for amt, precursor in zip(precursor_amts, precursors):
        if np.isclose(amt, 0.0):
            continue
        assert amt > 0, f"Amount of precursor {precursor} is negative!"

        p_thermo = MPInterpolatedMaterial.from_material_dict({'thermo': [{'formula': precursor}]})
        species.append(MaterialWithEnergy(
            thermo=p_thermo, composition=p_thermo.composition,
            is_target=False, side='reactant', amt=amt))

    # Gas compounds
    for amt, gas in zip(open_comp_amts, open_comp):
        if np.isclose(amt, 0.0):
            continue
        assert gas in fugacity, f'{gas} is a gas, but there is no fugacity data for it!'

        species.append(MaterialWithEnergy(
            thermo=GasMaterial(gas.copy(), fugacity[gas]), composition=gas.copy(),
            is_target=False, side='product' if amt < 0 else 'reactant', amt=abs(amt)))

    return ReactionEntry(
        k=k,
        reaction_string=equation,
        exp_t=row.temperature_Kelvin - 273.15,
        # Forgot to add this subtraction at the first time. Turned out out-of-the-sample predictions
        # using DMM dataset is so off by around 200 degrees. Was so confused about this mysterious
        # offset. But finally found the bug by looking at the distributions, which reminded me how
        # the offset of around 200 degrees came into play. Cheers!
        exp_time=9e99,
        reactions=[
            ReactionEnergies(target=target_thermo.composition, species=species, vars_sub={})
        ])


def from_pcd_entry_safe(row, k):
    try:
        return k, from_pcd_entry(row, k)
    except:
        return k, None


def pcd_to_reactions():
    argparser = ArgumentParser(description='Script to compute thermo cascades.')

    argparser.add_argument('--output-fn', '-o', type=str, default='PCD_Reactions.pypickle',
                           help='Output filename.')

    args = argparser.parse_args()

    pcd_dataset = synthesis_dataset.PCDDataset.get_PCD_data()
    pcd_oxides = pcd_dataset.loc[pcd_dataset.path.apply(lambda x: x.startswith('PCDBalanced/oxides'))]

    reactions = {}

    with Pool(processes=cpu_count()) as pool:
        for k, data in tqdm(pool.starmap(
                from_pcd_entry_safe,
                [(row, str(row_id)) for row_id, row in pcd_oxides.iterrows()]), total=len(pcd_oxides)):

            if data:
                reactions[k] = data

    with open(args.output_fn, 'wb') as f:
        pickle.dump(reactions, f)


if __name__ == '__main__':
    pcd_to_reactions()
