"""Process text-mined dataset into reactions."""
import os
import pickle
from argparse import ArgumentParser

from pymongo import MongoClient
from tqdm import tqdm

from s4.tmr.entry import from_reactions_multiprocessing


def download_reaction():
    """Download reactions from SynPro database."""

    print('Downloading reactions from SynPro')
    db = MongoClient('synthesisproject.lbl.gov').SynPro
    db.authenticate(
        'haoyan.huo@lbl.gov',
        '***REDACTED***'
    )
    col = db.Reactions_Solid_State_legacy_v14_2

    all_reactions = list(tqdm(col.find(), total=col.count_documents({})))
    data = {str(i): item for i, item in enumerate(all_reactions)}
    print('Downloaded %d reactions from SynPro' % len(data))
    return data


def load_dataset(possible_pickle_fn):
    """Load dataset or download from database."""
    if not os.path.exists(possible_pickle_fn):
        print('The pickle file %s does not exist, try to download from database...' % possible_pickle_fn)
        data = download_reaction()
        with open(possible_pickle_fn, 'wb') as f:
            pickle.dump(data, f)
    else:
        print('Loading recipes from %s' % possible_pickle_fn)
        with open(possible_pickle_fn, 'rb') as f:
            data = pickle.load(f)

    print('Found', len(data), 'Reactions')
    return data


def thermonize_reactions(reactions, using_mp=True, using_freed=True, processes=None):
    # Remove all oxygen del_O
    for k, d in reactions.items():
        if d['target']['thermo'] is not None:
            d['target']['thermo'] = list(filter(
                lambda x: not (x['amts_vars'] or {}).get('del_O', 0.0) > 0.0,
                d['target']['thermo']
            ))

    if using_mp:
        reaction_data_mp = from_reactions_multiprocessing(
            reactions, list(reactions), processes=processes,
            override_fugacity={}, return_errors=True)
        print('Thermonized', len(reaction_data_mp), 'reactions using MP data.')
    else:
        reaction_data_mp = None

    if using_freed:
        reaction_data_freed = from_reactions_multiprocessing(
            reactions, list(reactions), processes=processes,
            override_fugacity={}, use_database='freed', return_errors=True)
        print('Thermonized', len(reaction_data_mp), 'reactions using FREED data.')
    else:
        reaction_data_freed = None

    return reaction_data_mp, reaction_data_freed


def thermonize_dmm():
    argparser = ArgumentParser(description='Script to thermonize recipes '
                                           '(associate recipes with thermo quantities).')

    argparser.add_argument('--download-fn', '-d', type=str,
                           default='Reactions_Solid_State_legacy_v14_2.pickle',
                           help='Reactions download filename.')
    argparser.add_argument('--mp-output', '-mp', type=str, default=None,
                           help='Filename of thermonized recipes using Materials Project data.')
    argparser.add_argument('--freed-output', '-freed', type=str, default=None,
                           help='Filename of thermonized recipes using FREED data.')
    argparser.add_argument('--processes', '-p', type=int, default=None,
                           help='Number of multiprocesses to use.')

    args = argparser.parse_args()

    reactions = load_dataset(args.download_fn)
    mp, freed = thermonize_reactions(reactions,
                                     args.mp_output is not None,
                                     args.freed_output is not None,
                                     args.processes)

    if args.mp_output:
        with open(args.mp_output, 'wb') as f:
            pickle.dump(mp, f)

    if args.freed_output:
        with open(args.freed_output, 'wb') as f:
            pickle.dump(freed, f)


if __name__ == '__main__':
    thermonize_dmm()
