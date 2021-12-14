# Steps here:
#
# - Find all good cascades:
#     1. Synthesis time is greater than 1 hour.
#     2. Synthesis temperature is greater than 500 degC.
#     3. Target composition is not interpolated.
# - Compute features

import pickle
from argparse import ArgumentParser
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict

import numpy
import pandas
from pymatgen import Composition as C
from tqdm import tqdm

from s4.data import open_data
from s4.ml.features import Featurizer
from s4.tmr.entry import ReactionEntry


def check_target(cascade_result, reaction):
    if len(cascade_result) == 0:
        return {'err': True, 'msg': 'Empty cascade'}

    final_products = list(cascade_result[-1]['current_vessel'].keys())
    if len(final_products) > 1:
        return {'err': True, 'msg': 'Compounds remaining'}

    tgt = C(final_products[0])
    comps = [tgt[x] for x in tgt]

    # Target phase not desired
    desired_target = reaction.species[0].composition
    desired_comps = [desired_target[x] for x in tgt]
    if len(tgt) != len(desired_target) or not numpy.all(numpy.isclose(desired_comps, comps)):
        return {'err': True, 'msg': 'Wrong target'}
    else:
        return {'err': False, 'msg': 'Success'}


def how_many_succeeded(cascade_result, reactions_data):
    """
    How to define success here:

    Recipe level: at least one success
    Reaction level:
    """
    successful_results = {}
    failed_results = {}

    for k in cascade_result:
        success, fail = [], []
        for i, steps in cascade_result[k]:
            check_result = check_target(steps, reactions_data[k].reactions[i])
            if check_result['err']:
                fail.append((i, check_result))
            else:
                success.append((i, check_result))

        if success:
            successful_results[k] = success
        if fail:
            failed_results[k] = fail

    print('=' * 40 + '\nAnalysis of cascade results (per recipe):')
    successful_sets = set(successful_results)
    failed_sets = set(failed_results)
    print('Successful: %d, failed: %d, joint (has both success/failure): %d' % (
        len(successful_sets), len(failed_sets), len(successful_sets & failed_sets)
    ))
    return successful_results, failed_results


def get_dmm_data():
    with open_data('20210506_reaction_cascades.pypickle.only_icsd', 'rb') as f:
        cascade_only_icsd = pickle.load(f)
        print(f'Loaded {len(cascade_only_icsd)} cascade data (only ICSD)')
    with open_data('20210506_reaction_cascades.pypickle.all_mp.T=@', 'rb') as f:
        cascade_data = pickle.load(f)
        print(f'Loaded {len(cascade_data)} cascade data')
    with open_data('20210506_reaction_cascades.pypickle.all_mp.T=1K', 'rb') as f:
        cascade_data_1kdeg = pickle.load(f)
        print(f'Loaded {len(cascade_data_1kdeg)} cascade data (not only ICSD, 1K degC)')
    with open_data('20210506_reaction_cascades.pypickle.all_mp.T=1050', 'rb') as f:
        cascade_data_1050deg = pickle.load(f)
        print(f'Loaded {len(cascade_data_1050deg)} cascade data (not only ICSD, 1050 degC)')

    with open_data('20210506_reaction_data_mp_all.pypickle', 'rb') as f:
        normalized_reactions = pickle.load(f)
        print(f'Loaded {len(normalized_reactions)} reaction data')
        normalized_reactions = {k: normalized_reactions[k] for k in cascade_data}

    with open_data('20210506_Reactions_Solid_State_legacy_v14_2.pypickle', 'rb') as f:
        tms_data = pickle.load(f)
        print(f'Loaded {len(tms_data)} TMS data')
        # Only keep two fields to save memory.
        tms_data = {
            k: {'doi': v['doi'], 'ext_paragraph': v['ext_paragraph']}
            for k, v in tms_data.items()
        }

    print('Only ICSD')
    _, _ = how_many_succeeded(cascade_only_icsd, normalized_reactions)

    print('All MP')
    good_results, failed_results = how_many_succeeded(cascade_data, normalized_reactions)
    good_cascade_data = {
        k: [(i, dict(cascade_data[k])[i])
            for i, _ in data] for k, data in good_results.items()}

    print('All MP, 1K deg C')
    good_results, _ = how_many_succeeded(cascade_data_1kdeg, normalized_reactions)
    good_cascade_data_1k = {
        k: [(i, dict(cascade_data_1kdeg[k])[i]) for i, _ in data]
        for k, data in good_results.items()}

    return {
        'tms_recipes': tms_data,
        'normalized_reactions': normalized_reactions,
        'cascade_data': good_cascade_data,
        'cascade_data_1k': good_cascade_data_1k,
        'cascade_data_1050': cascade_data_1050deg,
    }


def load_any_data(normalized_reactions_fn, text_mined_recipes_file, cascades_file):
    with open(cascades_file, 'rb') as f:
        cascade_data = pickle.load(f)
        cascade_data = {
            key: [
                (i, data['@']['cascades'])
                for i, data in val if not data['@']['error']
            ]
            for key, val in cascade_data.items()
        }

    with open(normalized_reactions_fn, 'rb') as f:
        normalized_reactions, errors = pickle.load(f)
        print(f'Loaded {len(normalized_reactions)} reaction data')
        normalized_reactions = {k: normalized_reactions[k] for k in cascade_data}

    if text_mined_recipes_file is None:
        tms_data = defaultdict(lambda: {'doi': 'n/a', 'ext_paragraph': ['']})
    else:
        with open(text_mined_recipes_file, 'rb') as f:
            tms_data = pickle.load(f)
            print(f'Loaded {len(tms_data)} TMS data')
            # Only keep two fields to save memory.
            tms_data = {
                k: {'doi': v['doi'], 'ext_paragraph': v['ext_paragraph']}
                for k, v in tms_data.items()
            }

    print('All MP')
    good_results, failed_results = how_many_succeeded(cascade_data, normalized_reactions)
    good_cascade_data = {
        k: [(i, dict(cascade_data[k])[i])
            for i, _ in data] for k, data in good_results.items()}

    return {
        'tms_recipes': tms_data,
        'normalized_reactions': normalized_reactions,
        'cascade_data': good_cascade_data,
    }


def generate_df_for_prediction(
        tms_recipes, normalized_reactions: Dict[str, ReactionEntry], cascade_data):
    featurizer = Featurizer()

    precursor_counter = Counter()
    with Pool(processes=cpu_count() - 2) as pool:
        jobs = []
        for k, recipe_cascades in cascade_data.items():
            recipe = normalized_reactions[k]
            if recipe.exp_time < 1 or recipe.exp_t < 500:
                continue

            for i, reaction_cascade in recipe_cascades:
                precursors = tuple(sorted(reaction_cascade[0]['previous_vessel']))
                precursor_counter[precursors] += 1
                jobs.append((
                    recipe.reactions[i], k, i,
                    recipe.exp_t, recipe.exp_time,
                    tms_recipes[k]
                ))

        training_data = []
        for row in tqdm(pool.imap_unordered(featurizer, jobs), desc='Featurizing', total=len(jobs)):
            if row:
                row.update({
                    'meta_precursor_freq': precursor_counter[row['meta_precursors']],
                })
                training_data.append(row)

    return pandas.DataFrame(training_data)


def main():
    argparser = ArgumentParser(description='Script to compute features for synthesis condition ML.')

    argparser.add_argument('--reactions-file', '-r', type=str, default=None,
                           help='Filename of the reactions.')
    argparser.add_argument('--text-mined-recipes-file', '-t', type=str, default=None,
                           help='Filename of the text-mined recipes.')
    argparser.add_argument('--cascade-file', '-c', type=str, default=None,
                           help='Filename prefix of the cascades.')
    argparser.add_argument('--output-fn', '-o', type=str, required=True,
                           help='Output filename.')

    args = argparser.parse_args()

    if args.reactions_file is None:
        print('Using data-mined materials synthesis dataset...')
        data = get_dmm_data()
    else:
        data = load_any_data(args.reactions_file, args.text_mined_recipes_file, args.cascade_file)

    frame = generate_df_for_prediction(**data)
    print(f'Computed {len(frame)} entries')

    with open(args.output_fn, 'wb') as f:
        print(f'Writing results to {args.output_fn}')
        pickle.dump(frame, f)


if __name__ == '__main__':
    main()
