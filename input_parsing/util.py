"""Util for feature generation."""

import numpy as np
from rdkit import Chem

ELEMENT_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
                'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                'In',
                'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo',
                'U',
                'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']

BOND_TYPE_TO_CHARGE = {
    Chem.rdchem.BondType.SINGLE: 2,
    Chem.rdchem.BondType.DOUBLE: 4,
    Chem.rdchem.BondType.TRIPLE: 6,
    Chem.rdchem.BondType.AROMATIC: 3
}

BOND_TYPE = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]

CHARGE_TO_BOND_TYPE = {charge: bond_type
                       for bond_type, charge in BOND_TYPE_TO_CHARGE.items()}

ATOM_FEATURE_DIM = 151
BOND_FEATURE_DIM = 11
MAX_NEIGHBOR_NUM = 10

ATOM_FEATURES_KEY = 'atom_features'
BOND_FEATURES_KEY = 'bond_features'
NEIGHBOR_ATOM_KEY = 'neighbor_atom'
NEIGHBOR_BOND_KEY = 'neighbor_bond'
NEIGHBOR_MASK_KEY = 'neighbor_mask'
REACTION_STR_KEY = 'reaction_str'
EDIT_STR_KEY = 'edit_str'

COMPONENT_MAP_KEY = 'comp_map'
COMPONENT_NUM_KEY = 'comp_num'

EDGE_DELTA_KEY = 'edge_delta'
H_DELTA_KEY = 'h_delta'
C_DELTA_KEY = 'c_delta'
OCTET_SUM_KEY = 'octet_sum'

FEATURE_KEY = 'features'
LABEL_KEY = 'labels'

EDGE_DELTA_VAL_LIST = np.array([-6, -4, -2, 0, 2, 4, 6])
H_DELTA_VAL_LIST = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
C_DELTA_VAL_LIST = np.array([-2, 0, 2, 4, 6])

CANDIDATE_RANKING_ATOM_FEATURE_DIM = 2 * ATOM_FEATURE_DIM + len(H_DELTA_VAL_LIST) + len(C_DELTA_VAL_LIST) + 1
CANDIDATE_RANKING_BOND_FEATURE_DIM = 2 * BOND_FEATURE_DIM + len(EDGE_DELTA_VAL_LIST) + 1

CANDIDATE_RANKING_LABEL_KEY = 'candidate_label'
CANDIDATE_STR_KEY = 'candidate_strs'
CANDIDATE_RANKED_KEY = 'candidate_rank'
CANDIDATE_TOP_RANKED_KEY = 'top_ranked_candidate'


def count_heavy_atom(reaction_str):
    return reaction_str.count(':')


def get_reactant_product_str(reaction_str):
    return reaction_str.split('>>')


def get_reactant_product_molecule(reaction_str):
    reactant_str, product_str = get_reactant_product_str(reaction_str)
    reactant_mol = Chem.MolFromSmiles(reactant_str)
    if reactant_mol is None:
        raise ValueError('Cannot parse the reactant SMILE to molecule:\n{}'.format(reactant_str))
    product_mol = Chem.MolFromSmiles(product_str)
    if product_mol is None:
        raise ValueError('Cannot parse the product SMILE to molecule:\n{}'.format(product_mol))

    return reactant_mol, product_mol


def get_reactant_mol_size_from_reaction_str(reaction_str):
    reactant_mol, _ = get_reactant_product_molecule(reaction_str)
    return reactant_mol.GetNumAtoms()


def idxfunc(atom):
    return atom.GetIntProp('molAtomMapNumber') - 1


def one_hot_encoding(val, one_hot_vals):
    return list(map(lambda s: val == s, one_hot_vals))


def get_reactant_mols(reactant_mol):
    return [Chem.MolFromSmiles(r_str) for r_str in Chem.MolToSmiles(reactant_mol).split('.')]


def get_reactant_atom_idx(reactant_mols, product_mol):
    reactant_am_nums = set()
    product_am_nums = {idxfunc(atom) for atom in product_mol.GetAtoms()}
    for mol in reactant_mols:
        mol_am_nums = {idxfunc(atom) for atom in mol.GetAtoms()}
        for am_num in mol_am_nums:
            if am_num in product_am_nums:
                reactant_am_nums.update(mol_am_nums)
                break
    return reactant_am_nums


def get_reactant_component_map(reactant_mols):
    comp_map = {}
    for i, mol in enumerate(reactant_mols):
        for atom in mol.GetAtoms():
            comp_map[idxfunc(atom)] = 1
    return {COMPONENT_MAP_KEY: comp_map,
            COMPONENT_NUM_KEY: len(reactant_mols)}


def _compare_molecule_by_smiles(target_mol, pred_mol, strip_atommap=False, verbose=False):
    if strip_atommap:
        # Copy Molecule So atom map isn't stripped via reference
        target_mol = Chem.Mol(target_mol)
        pred_mol = Chem.Mol(pred_mol)
        if verbose:
            print("Stripping Away Atom Map Numbers")
        for atom in target_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        for atom in pred_mol.GetAtoms():
            atom.SetAtomMapNum(0)

    # Get Smiles Sets
    target_smi = Chem.MolToSmiles(target_mol)
    target_smi_set = set(target_smi.split("."))
    pred_smi = Chem.MolToSmiles(pred_mol)
    pred_smi_set = set(pred_smi.split("."))

    # Pred smi set should be superset of target
    if pred_smi_set >= target_smi_set:
        if verbose:
            print("Smiles Match!")
        return True
    elif strip_atommap is False:
        if verbose:
            print("No Match with AtomMap still in smiles")
        return _compare_molecule_by_smiles(target_mol, pred_mol, strip_atommap=True, verbose=verbose)
    else:
        if verbose:
            print("No Match even after stripping AtomMap numbers")
        return False

def is_same_molecule(target_mol, pred_mol, allow_switch=False, sanitize=True, verbose=False):
    pred_n = pred_mol.GetNumAtoms()
    target_n = target_mol.GetNumAtoms()

    if pred_n < target_n:
        if allow_switch:
            target_mol, pred_mol = pred_mol, target_mol
        else:
            raise ValueError("Predicted Molecule have at least as many atoms as Target Molecule\n"
                             "Pred Atoms = %d, Target Atoms = %d" % (pred_n, target_n))

    if sanitize:
        try:
            Chem.SanitizeMol(target_mol)
            Chem.SanitizeMol(pred_mol)
            Chem.Kekulize(target_mol, clearAromaticFlags=False)
            Chem.Kekulize(pred_mol, clearAromaticFlags=False)
            Chem.SanitizeMol(target_mol)
            Chem.SanitizeMol(pred_mol)
        except ValueError:
            return None

    if _compare_molecule_by_smiles(target_mol, pred_mol, verbose=verbose):
        return True
    else:
        return False
