"""Util for symbolic inference."""

import numpy as np
from rdkit import Chem

from input_parsing.util import idxfunc, count_heavy_atom, is_same_molecule

SOFTEN_EPSILON = 1e-8
SELECTION_EPSILON = 0.1

EDGE_DELTA_VAR_NAME_HEADER = 'edge-delta'
H_DELTA_VAR_NAME_HEADER = 'h-delta'
C_DELTA_VAR_NAME_HEADER = 'c-delta'

GUROBI_KEY = 'Gurobi'
COUNT_KEY = 'found_in_count'
TIME_KEY = 'found_in_second'
FOUND_KEY = 'found_in_solution'

SAMPLE_SOLUTION_MOL_KEY = 'solution_mol'
SAMPLE_SOLUTION_VAL_KEY = 'solution_objective_val'
SAMPLE_SOLUTION_OUTPUT_PKL_FILE_KEY = 'solutions.pkl'
SAMPLE_SOLUTION_OUTPUT_TXT_FILE_KEY = 'solutions.txt'
SAMPLE_SOLUTION_OUTPUT_TXT_DIVIDER = '=='

EDGE_CALIBRATION_KEY = 'edge_calibration'
H_CALIBRATION_KEY = 'h_calibration'


def bond_idx_tuple(bond):
    return tuple(sorted([idxfunc(bond.GetBeginAtom()), idxfunc(bond.GetEndAtom())]))


def soften_matrix(mat):
    mat = mat + SOFTEN_EPSILON
    mat = mat / np.sum(mat, axis=-1, keepdims=True)
    return mat


def parse_top_k_arg(top_k_str):
    topks = top_k_str.split(',')
    topks = [topk.strip() for topk in topks]
    try:
        return sorted([int(topk) for topk in topks])
    except:
        raise ValueError('Cannot parse --topk as: {}'.format(top_k_str))


def find_primary_product(product_mol_smile):
    component_smiles = product_mol_smile.split('.')
    component_heavy_atom_counts = [count_heavy_atom(component_smile) for component_smile in component_smiles]
    major_product_idx = np.argsort(component_heavy_atom_counts)[-1]
    return Chem.MolFromSmiles(component_smiles[major_product_idx])


def find_primary_product_using_reactant_idx(product_mol_smile, reactant_idx):
    def count_reactant_atom(mol_str):
        count = 0
        mol = Chem.MolFromSmiles(mol_str)
        for atom in mol.GetAtoms():
            if idxfunc(atom) in reactant_idx:
                count += 1
        return count

    component_smiles = product_mol_smile.split('.')
    component_heavy_atom_counts = [count_reactant_atom(component_smile) for component_smile in component_smiles]
    major_product_idx = np.argsort(component_heavy_atom_counts)[-1]
    return Chem.MolFromSmiles(component_smiles[major_product_idx])


def unique_primary_product_idxs(primary_product_mols):
    included_primary_products = []
    unique_idxs = []
    for idx, primary_product_mol in enumerate(primary_product_mols):
        if primary_product_mol is None:
            continue
        included = False
        for included_primary_product in included_primary_products:
            if is_same_molecule(included_primary_product, primary_product_mol,
                                allow_switch=True, ):
                included = True
                break
        if not included:
            included_primary_products.append(primary_product_mol)
            unique_idxs.append(idx)
    return unique_idxs


def unique_primary_product_idxs_v2(primary_product_mols):
    def strip_atom_map(mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    for primary_product_mol in primary_product_mols:
        strip_atom_map(primary_product_mol)
    primary_product_smiles = [Chem.MolToSmiles(primary_product_mol) for primary_product_mol in primary_product_mols]
    included_primary_products = set()
    unique_idxs = []
    for idx, primary_product_smile in enumerate(primary_product_smiles):
        if primary_product_smile not in included_primary_products:
            included_primary_products.add(primary_product_smile)
            unique_idxs.append(idx)
    return unique_idxs


def filter_solutions_with_same_primary_product(product_mols, reactant_idx=None):
    if reactant_idx is None:
        primary_products = [find_primary_product(Chem.MolToSmiles(product_mol[SAMPLE_SOLUTION_MOL_KEY]))
                            for product_mol in product_mols]
    else:
        primary_products = [
            find_primary_product_using_reactant_idx(Chem.MolToSmiles(product_mol[SAMPLE_SOLUTION_MOL_KEY]),
                                                    reactant_idx)
            for product_mol in product_mols]
    unique_idxs = unique_primary_product_idxs(primary_products)
    return [product_mols[unique_idx] for unique_idx in unique_idxs]


def smooth_calibrate(probas):
    def _calibrate(proba):
        return max(min(1, _scale(proba)), 0)

    def _scale(x):
        if x >= 1 - 1e-9 or x <= 1e-9:
            return x
        return np.log(x / (1 - x)) / 50 + 0.5

    probas = [_calibrate(proba) for proba in probas]
    probas = np.array(probas) / np.sum(probas)
    return probas


def no_calibration(probas):
    return probas


def get_atom_octet_charge_change(atom):
    symbol = atom.GetSymbol()

    # Common Exceptions
    if symbol in {"P", "Li"}:
        ub, lb = 4, -4
    elif symbol == "S":
        ub, lb = 8, -8

    # Aromatic Exceptions
    elif atom.GetIsAromatic():
        if symbol == "C" and has_aromatic_nonring(atom):
            ub, lb = 2, -2
        elif symbol == "N":
            ub, lb = 2, -2
        else:
            ub, lb = 0, 0

    # Less Common Exceptions
    elif symbol in {"Ag", "Al", "Au", "B", "Ba", "Ca", "Cd", "Co", "Ti", "Zr"}:
        ub, lb = None, None
    elif symbol in {"I", "Sn"}:
        ub, lb = 4, -8
    else:
        ub, lb = 0, 0
    return ub, lb


def has_aromatic_nonring(atom):
    for bond in atom.GetBonds():
        aromatic = bond.GetBondType() == Chem.rdchem.BondType.AROMATIC
        other_atom = bond.GetOtherAtom(atom)
        if not aromatic and not other_atom.IsInRing() and other_atom.GetSymbol() != "C":
            return True
    return False
