"""Generate reactivity prediction label."""

import numpy as np

from input_parsing.util import idxfunc, BOND_TYPE_TO_CHARGE, EDGE_DELTA_KEY, H_DELTA_KEY, \
    C_DELTA_KEY, OCTET_SUM_KEY, get_reactant_atom_idx, get_reactant_mols


def _get_edge_delta_label(reactant_mol, product_mol):
    num_atom = reactant_mol.GetNumAtoms()

    reactant_edge_charge = np.zeros((num_atom, num_atom), dtype=np.int)
    product_edge_charge = np.zeros_like(reactant_edge_charge)

    for bond in reactant_mol.GetBonds():
        begin_atom_idx = idxfunc(bond.GetBeginAtom())
        end_atom_idx = idxfunc(bond.GetEndAtom())
        bond_charge = BOND_TYPE_TO_CHARGE[bond.GetBondType()]
        reactant_edge_charge[begin_atom_idx, end_atom_idx] = bond_charge
        reactant_edge_charge[end_atom_idx, begin_atom_idx] = bond_charge

    for bond in product_mol.GetBonds():
        begin_atom_idx = idxfunc(bond.GetBeginAtom())
        end_atom_idx = idxfunc(bond.GetEndAtom())
        bond_charge = BOND_TYPE_TO_CHARGE[bond.GetBondType()]
        product_edge_charge[begin_atom_idx, end_atom_idx] = bond_charge
        product_edge_charge[end_atom_idx, begin_atom_idx] = bond_charge

    return product_edge_charge - reactant_edge_charge


def _get_hydrogen_delta_label(reactant_mol, product_mol):
    num_atom = reactant_mol.GetNumAtoms()

    reactant_h_count = np.zeros(num_atom, dtype=np.int)
    product_h_count = np.zeros_like(reactant_h_count)

    for atom in reactant_mol.GetAtoms():
        reactant_h_count[idxfunc(atom)] = atom.GetTotalNumHs()

    for atom in product_mol.GetAtoms():
        product_h_count[idxfunc(atom)] = atom.GetTotalNumHs()

    return (product_h_count - reactant_h_count) * 2


def _get_charge_delta_label(reactant_mol, product_mol):
    num_atom = reactant_mol.GetNumAtoms()

    reactant_c_count = np.zeros(num_atom, dtype=np.int)
    product_c_count = np.zeros_like(reactant_c_count)

    for atom in reactant_mol.GetAtoms():
        reactant_c_count[idxfunc(atom)] = atom.GetFormalCharge()

    for atom in product_mol.GetAtoms():
        product_c_count[idxfunc(atom)] = atom.GetFormalCharge()

    return (product_c_count - reactant_c_count) * -2


def get_delta_labels(reactant_mol, product_mol):
    product_atom_idx = {idxfunc(atom) for atom in product_mol.GetAtoms()}
    reactant_atom_idx = get_reactant_atom_idx(get_reactant_mols(reactant_mol), product_mol)

    edge_deltas = _get_edge_delta_label(reactant_mol, product_mol)
    h_deltas = _get_hydrogen_delta_label(reactant_mol, product_mol)
    c_deltas = _get_charge_delta_label(reactant_mol, product_mol)

    num_atom = reactant_mol.GetNumAtoms()
    octet_sum = np.zeros(num_atom, dtype=np.int)
    for idx in range(num_atom):

        for idx_other in range(num_atom):
            if idx not in product_atom_idx and idx_other not in product_atom_idx:
                edge_deltas[idx, idx_other] = edge_deltas[idx_other, idx] = 0

        if idx not in product_atom_idx and idx in reactant_atom_idx:
            # assume h on break
            h_deltas[idx] = -np.sum(edge_deltas[idx])
            c_deltas[idx] = 0
        elif idx not in product_atom_idx:
            h_deltas[idx] = 0
            c_deltas[idx] = 0

        octet_sum[idx] = np.sum(edge_deltas[idx]) + h_deltas[idx] + c_deltas[idx]

    return {EDGE_DELTA_KEY: edge_deltas,
            H_DELTA_KEY: h_deltas,
            C_DELTA_KEY: c_deltas,
            OCTET_SUM_KEY: octet_sum}
