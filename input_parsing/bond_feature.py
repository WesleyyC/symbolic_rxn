"""Generate bond features."""

import numpy as np
from rdkit import Chem

from input_parsing.util import BOND_FEATURE_DIM, MAX_NEIGHBOR_NUM, idxfunc, one_hot_encoding, BOND_TYPE_TO_CHARGE, \
    BOND_FEATURES_KEY, NEIGHBOR_ATOM_KEY, NEIGHBOR_BOND_KEY, NEIGHBOR_MASK_KEY, COMPONENT_MAP_KEY, COMPONENT_NUM_KEY


def _get_bond_features(bond):
    bt = bond.GetBondType()
    bt_k = one_hot_encoding(bt, [Chem.rdchem.BondType.SINGLE,
                                 Chem.rdchem.BondType.DOUBLE,
                                 Chem.rdchem.BondType.TRIPLE,
                                 Chem.rdchem.BondType.AROMATIC])
    return np.array(bt_k + [bond.GetIsConjugated(), bond.IsInRing(), BOND_TYPE_TO_CHARGE[bt]], dtype=np.float)


def get_mol_bond_features(reactant_mol, num_atom=None, reactant_atom_idx=None, reactant_component_map=None):
    if num_atom is None:
        num_atom = reactant_mol.GetNumAtoms()

    bond_features = np.zeros((num_atom, num_atom, BOND_FEATURE_DIM), dtype=np.float)
    neighbor_atom = np.zeros((num_atom, MAX_NEIGHBOR_NUM), dtype=np.int)
    neighbor_bond = np.zeros((num_atom, MAX_NEIGHBOR_NUM, 2), dtype=np.int)
    neighbor_mask = np.zeros_like(neighbor_atom)
    neighbor_num = np.zeros(num_atom, dtype=np.int)

    for bond in reactant_mol.GetBonds():
        begin_atom_idx = idxfunc(bond.GetBeginAtom())
        end_atom_idx = idxfunc(bond.GetEndAtom())

        neighbor_atom[begin_atom_idx, neighbor_num[begin_atom_idx]] = end_atom_idx
        neighbor_atom[end_atom_idx, neighbor_num[end_atom_idx]] = begin_atom_idx
        neighbor_bond[begin_atom_idx, neighbor_num[begin_atom_idx]] = [begin_atom_idx, end_atom_idx]
        neighbor_bond[end_atom_idx, neighbor_num[end_atom_idx]] = [end_atom_idx, begin_atom_idx]
        neighbor_mask[begin_atom_idx, neighbor_num[begin_atom_idx]] = 1
        neighbor_mask[end_atom_idx, neighbor_num[end_atom_idx]] = 1
        neighbor_num[begin_atom_idx] += 1
        neighbor_num[end_atom_idx] += 1

        bond_feature = _get_bond_features(bond)

        if reactant_atom_idx is not None and (begin_atom_idx in reactant_atom_idx or end_atom_idx in reactant_atom_idx):
            bond_feature = np.append(bond_feature, 1)
        else:
            bond_feature = np.append(bond_feature, 0)

        # bond exists
        bond_feature = np.append(bond_feature, 1)

        bond_features[begin_atom_idx, end_atom_idx, :BOND_FEATURE_DIM - 2] = bond_feature
        bond_features[end_atom_idx, begin_atom_idx, :BOND_FEATURE_DIM - 2] = bond_feature

    for i in range(num_atom):
        for j in range(num_atom):
            if reactant_component_map is not None and reactant_component_map[COMPONENT_MAP_KEY][i] == \
                    reactant_component_map[COMPONENT_MAP_KEY][j]:
                bond_features[i, j, -2] = bond_features[j, i, -2] = 1
            else:
                bond_features[i, j, -2] = bond_features[j, i, -2] = 0

            if reactant_component_map is not None and reactant_component_map[COMPONENT_NUM_KEY] > 1:
                bond_features[i, j, -1] = bond_features[j, i, -1] = 1
            else:
                bond_features[i, j, -1] = bond_features[j, i, -1] = 0

    return {BOND_FEATURES_KEY: bond_features,
            NEIGHBOR_ATOM_KEY: neighbor_atom,
            NEIGHBOR_BOND_KEY: neighbor_bond,
            NEIGHBOR_MASK_KEY: neighbor_mask}
