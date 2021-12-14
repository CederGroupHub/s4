import logging
import os
import pickle
import sys
import traceback
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from s4.cascade.analysis import compute_cascade
from s4.data import open_data

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s]: %(message)s')


def _init_subprocess(o_icsd, use_temperature):
    _init_subprocess.only_icsd = o_icsd
    _init_subprocess.use_temperature = use_temperature
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')


def _compute_cascade_for_one(args):
    k, d = args
    temperatures = _init_subprocess.use_temperature

    results = []
    for j, reaction in enumerate(d.reactions):
        per_reaction = {}
        for i, temp in enumerate([d.exp_t] + temperatures):
            furnace = [temp + 273.15] * 10

            try:
                step_info = compute_cascade(reaction, furnace, only_icsd=_init_subprocess.only_icsd)
                result = {
                    'error': False,
                    'cascades': step_info,
                }
            except Exception as exp:
                trace = traceback.format_exc()
                result = {
                    'error': True,
                    'traceback': trace,
                    'exception': type(exp),
                    'exception_desc': str(exp)
                }

            if i == 0:
                per_reaction['@'] = result
            else:
                per_reaction[temp] = result

        results.append((j, per_reaction))

    return k, results


def parallel_apply(reactions, only_icsd, use_temperature, what_is_it):
    with Pool(processes=cpu_count(),
              initializer=_init_subprocess,
              initargs=(only_icsd, use_temperature)) as pool:
        cascade_data = {
            k: results
            for k, results in tqdm(
                pool.imap_unordered(
                    _compute_cascade_for_one, reactions.items(),
                    chunksize=max(1, min(len(reactions) // cpu_count(), 16))),
                desc=what_is_it,
                total=len(reactions))
            if results
        }
        return cascade_data


def run_with_arguments():
    argparser = ArgumentParser(description='Script to compute thermo cascades.')

    argparser.add_argument('--reactions-file', '-r', type=str,
                           help='Filename of the reactions.')
    argparser.add_argument('--use-dmm', '-dmm', action='store_true', default=False,
                           help='Use data-mined materials synthesis dataset.')
    argparser.add_argument('--output-fn', '-o', type=str, required=True,
                           help='Output filename.')
    argparser.add_argument('--cascade-temperatures', '-t', type=str, default='1000',
                           help='Temperatures at which the cascade will be run.')
    argparser.add_argument('--only-icsd', '-a', action='store_true', default=False,
                           help='Only use ICSD structures.')

    args = argparser.parse_args()
    assert args.reactions_file is not None or args.use_dmm

    if args.use_dmm:
        print('Using data-mined materials synthesis dataset...')
        with open_data('20210506_reaction_data_mp_all.pypickle', 'rb') as f:
            reactions, errors = pickle.load(f)
    else:
        print('Loading dataset from %s' % args.reactions_file)
        with open(args.reactions_file, 'rb') as f:
            reactions, errors = pickle.load(f)

    print('Found', len(reactions), 'recipes')
    reactions = {k: v for k, v in reactions.items() if v.exp_t >= 500}
    print('Found', len(reactions), 'valid recipes with temperature above 500 degC')

    temperatures = eval(args.cascade_temperatures)

    print('Computing cascades at the specified temperatures: %r...' % temperatures)
    cascade_data = parallel_apply(
        reactions, only_icsd=args.only_icsd, use_temperature=temperatures, what_is_it='Cascades')
    with open(f'{args.output_fn}', 'wb') as f:
        pickle.dump(cascade_data, f)


if __name__ == '__main__':
    run_with_arguments()
