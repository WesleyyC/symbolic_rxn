"""Running symbolic inference with evaluation."""

import argparse
from time import perf_counter

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit import rdBase

from input_parsing.util import get_reactant_product_molecule, is_same_molecule, get_reactant_atom_idx, get_reactant_mols
from octet_sampling.gurobi_sampler import run_gurobi_sampler
from octet_sampling.util import parse_top_k_arg, GUROBI_KEY, COUNT_KEY, TIME_KEY, SAMPLE_SOLUTION_MOL_KEY, \
    SAMPLE_SOLUTION_VAL_KEY, find_primary_product_using_reactant_idx
from reactivity_prediction.io import OUTPUT_REACTION_STR_KEY, load_delta_pred_list

parser = argparse.ArgumentParser(description='Running Sampler')
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--topk', type=str, default='1,3,5,10,20')
parser.add_argument('--log_interval', type=int, default=500)
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()

top_ks = parse_top_k_arg(args.topk)

if not args.verbose:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)
    rdBase.DisableLog('rdApp.error')

RESULT = {GUROBI_KEY: {COUNT_KEY: [],
                       TIME_KEY: []}}


def log_result(idx):
    print('=' * 32 + ' #{} of Items '.format(idx) + '=' * 32)
    for sampler, sampler_result in RESULT.items():
        log_str = '{}: '.format(sampler)
        sampler_time = sampler_result[TIME_KEY]
        sampler_count = np.array(sampler_result[COUNT_KEY])
        for top_k in top_ks:
            val = (sampler_count >= 0) * (sampler_count < top_k)
            val = np.mean(val)
            log_str += 'Top{}: {:.3f} '.format(top_k, val)
        log_str += 'Average Time: {:.3f}s'.format(np.mean(sampler_time))
        print(log_str)


def eval_gurobi_sampler(delta_pred_list):
    sample_solution_list = []

    for idx, delta_pred in enumerate(delta_pred_list):
        reaction_str = delta_pred[OUTPUT_REACTION_STR_KEY]
        reactant_mol, product_mol = get_reactant_product_molecule(reaction_str)
        reactant_atom_idx = get_reactant_atom_idx(get_reactant_mols(reactant_mol), product_mol)

        time_lapse = perf_counter()
        solutions = run_gurobi_sampler(delta_pred,
                                       num_candidates=max(top_ks),
                                       verbose=args.verbose)
        time_lapse = perf_counter() - time_lapse

        duplications = set()
        alt_solutions = []
        for solution in solutions:
            solution_mol = solution[SAMPLE_SOLUTION_MOL_KEY]
            solution_mol = find_primary_product_using_reactant_idx(Chem.MolToSmiles(solution_mol), reactant_atom_idx)
            for atom in solution_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            solution_smi = Chem.MolToSmiles(solution_mol)
            if solution_smi in duplications:
                continue
            else:
                duplications.add(solution_smi)
                alt_solutions.append(solution)
        solutions = alt_solutions

        found_in_count = -1
        solution_strs = []
        solution_obj_vals = []
        for solution_idx, solution in enumerate(solutions):
            solution_mol = solution[SAMPLE_SOLUTION_MOL_KEY]
            solution_strs.append(Chem.MolToSmiles(solution_mol))
            solution_obj_vals.append(solution[SAMPLE_SOLUTION_VAL_KEY])
            if found_in_count < 0 and is_same_molecule(product_mol, solution_mol):
                found_in_count = solution_idx

        delta_pred[SAMPLE_SOLUTION_MOL_KEY] = solution_strs
        delta_pred[SAMPLE_SOLUTION_VAL_KEY] = solution_obj_vals

        RESULT[GUROBI_KEY][COUNT_KEY].append(found_in_count)
        RESULT[GUROBI_KEY][TIME_KEY].append(time_lapse)

        sample_solution_list.append(delta_pred)

        if args.verbose:
            print('#{}: {}'.format(idx, found_in_count))

        if idx % args.log_interval == 0:
            log_result(idx)

    log_result(len(sample_solution_list))


if __name__ == '__main__':
    delta_pred_list = load_delta_pred_list(args.input)
    eval_gurobi_sampler(delta_pred_list)
