"""Parse SMILES string into numpy features."""

import numpy as np
from rdkit import Chem

from input_parsing.atom_feature import get_mol_atom_features
from input_parsing.bond_feature import get_mol_bond_features
from input_parsing.delta_label import get_delta_labels
from input_parsing.util import get_reactant_mols, get_reactant_component_map, get_reactant_product_molecule, \
    get_reactant_atom_idx, FEATURE_KEY, LABEL_KEY, REACTION_STR_KEY, EDIT_STR_KEY, ATOM_FEATURES_KEY, BOND_FEATURES_KEY


def _get_reactivity_prediction_features(reactant_mol, product_mol):
    reactant_mols = get_reactant_mols(reactant_mol)
    reactant_atom_idx = get_reactant_atom_idx(reactant_mols, product_mol)
    reactant_component_map = get_reactant_component_map(reactant_mols)
    atom_features = get_mol_atom_features(reactant_mol,
                                          reactant_atom_idx=reactant_atom_idx)
    bond_features = get_mol_bond_features(reactant_mol,
                                          reactant_atom_idx=reactant_atom_idx,
                                          reactant_component_map=reactant_component_map)
    return {**atom_features, **bond_features}


def _get_reactivity_prediction_labels(reactant_mol, product_mol):
    return get_delta_labels(reactant_mol, product_mol)


def get_reactivity_prediction_features_labels(reaction_str, edit_str=None):
    reactant_mol, product_mol = get_reactant_product_molecule(reaction_str)
    features = _get_reactivity_prediction_features(reactant_mol, product_mol)
    labels = _get_reactivity_prediction_labels(reactant_mol, product_mol)

    if edit_str is None:
        edit_str = ''

    return {
        REACTION_STR_KEY: reaction_str,
        EDIT_STR_KEY: edit_str,
        FEATURE_KEY: features,
        LABEL_KEY: labels
    }


def _pad_features_with_val(features, val):
    shape = list(features.shape)
    shape[-1] = 1
    val_features = np.ones(tuple(shape)) * val
    features = np.concatenate((features, val_features), axis=-1)
    return features


def get_candidate_ranking_features(reaction_str,
                                   candidate_str,
                                   candidate_val,
                                   edge_delta_pred,
                                   h_delta_pred,
                                   c_delta_pred):
    reactant_mol, product_mol = get_reactant_product_molecule(reaction_str)

    num_atom = reactant_mol.GetNumAtoms()

    reactant_mols = get_reactant_mols(reactant_mol)
    reactant_atom_idx = get_reactant_atom_idx(reactant_mols, product_mol)
    reactant_component_map = get_reactant_component_map(reactant_mols)
    reactant_atom_features = get_mol_atom_features(reactant_mol,
                                                   reactant_atom_idx=reactant_atom_idx)
    reactant_bond_features = get_mol_bond_features(reactant_mol,
                                                   reactant_atom_idx=reactant_atom_idx,
                                                   reactant_component_map=reactant_component_map)

    candidate_mol = Chem.MolFromSmiles(candidate_str)
    candidate_atom_features = get_mol_atom_features(candidate_mol,
                                                    num_atom=num_atom)
    candidate_bond_features = get_mol_bond_features(candidate_mol,
                                                    num_atom=num_atom)

    candidate_atom_features[ATOM_FEATURES_KEY] = np.concatenate(
        (candidate_atom_features[ATOM_FEATURES_KEY],
         reactant_atom_features[ATOM_FEATURES_KEY],
         h_delta_pred,
         c_delta_pred),
        axis=-1)
    candidate_bond_features[BOND_FEATURES_KEY] = np.concatenate(
        (candidate_bond_features[BOND_FEATURES_KEY],
         reactant_bond_features[BOND_FEATURES_KEY],
         edge_delta_pred),
        axis=-1
    )

    candidate_atom_features[ATOM_FEATURES_KEY] = _pad_features_with_val(candidate_atom_features[ATOM_FEATURES_KEY],
                                                                        candidate_val)
    candidate_bond_features[BOND_FEATURES_KEY] = _pad_features_with_val(candidate_bond_features[BOND_FEATURES_KEY],
                                                                        candidate_val)

    return {**candidate_atom_features, **candidate_bond_features}
