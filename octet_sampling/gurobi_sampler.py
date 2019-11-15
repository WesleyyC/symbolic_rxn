"""Symbolic inference implementation in Gurobi."""

import numpy as np
from gurobi import *
from rdkit import Chem

from input_parsing.util import get_reactant_product_molecule, get_reactant_atom_idx, get_reactant_mols, idxfunc, \
    BOND_TYPE_TO_CHARGE, EDGE_DELTA_VAL_LIST, H_DELTA_VAL_LIST, C_DELTA_VAL_LIST, CHARGE_TO_BOND_TYPE
from octet_sampling.util import bond_idx_tuple, soften_matrix, EDGE_DELTA_VAR_NAME_HEADER, H_DELTA_VAR_NAME_HEADER, \
    C_DELTA_VAR_NAME_HEADER, SELECTION_EPSILON, SAMPLE_SOLUTION_MOL_KEY, SAMPLE_SOLUTION_VAL_KEY, \
    smooth_calibrate, no_calibration, \
    EDGE_CALIBRATION_KEY, H_CALIBRATION_KEY, get_atom_octet_charge_change
from reactivity_prediction.io import OUTPUT_REACTION_STR_KEY, OUTPUT_EDGE_DELTA_KEY, OUTPUT_H_DELTA_KEY, \
    OUTPUT_C_DELTA_KEY

INCREMENT_FACTOR = 3
MAX_INCREMENT_ROUND = 4
CLIP_PROB_THRESHOLD = 1 - 1e-9


class GurobiSampler:

    def __init__(self, delta_pred,
                 num_candidates=10,
                 calibration=(EDGE_CALIBRATION_KEY),
                 soften=True,
                 octet_rule=True,
                 verbose=False):
        self.reaction_str = delta_pred[OUTPUT_REACTION_STR_KEY]
        self.edge_delta_pred = delta_pred[OUTPUT_EDGE_DELTA_KEY]
        self.c_delta_pred = delta_pred[OUTPUT_C_DELTA_KEY]
        self.h_delta_pred = delta_pred[OUTPUT_H_DELTA_KEY]
        if soften:
            self.edge_delta_pred = soften_matrix(self.edge_delta_pred)
            self.c_delta_pred = soften_matrix(self.c_delta_pred)
            self.h_delta_pred = soften_matrix(self.h_delta_pred)
        self.num_candidates = num_candidates
        self.edge_coefficient = 5.0
        self.h_coefficient = 1.0
        self.c_coefficient = 1.0
        self.octet_rule = octet_rule
        self.edge_calibration_fn = no_calibration
        self.h_calibration_fn = no_calibration
        if EDGE_CALIBRATION_KEY in calibration:
            self.edge_calibration_fn = smooth_calibrate
        if H_CALIBRATION_KEY in calibration:
            self.h_calibration_fn = smooth_calibrate
        self.verbose = verbose

        self.reactant_mol, self.product_mol = get_reactant_product_molecule(self.reaction_str)
        Chem.SanitizeMol(self.reactant_mol)
        Chem.Kekulize(self.reactant_mol, clearAromaticFlags=True)

        self.n_atom = self.reactant_mol.GetNumAtoms()

        self.reactant_atom_map = {idxfunc(atom): atom for atom in self.reactant_mol.GetAtoms()}
        self.reactant_bond_map = {bond_idx_tuple(bond): bond for bond in self.reactant_mol.GetBonds()}

        self.reactant_atom_idx = get_reactant_atom_idx(get_reactant_mols(self.reactant_mol), self.product_mol)

        self.idx_to_delta_vars = {}
        self.reaction_center_delta_vars = []
        self.model_objective = []

        self.model = Model('Gurobi Sampler for Octet Sampling')
        if not self.verbose:
            self.model.setParam(GRB.Param.OutputFlag, 0)

        self._set_variables()

        self._set_constraints()

        self._set_model_objective()

        self._set_model_param()

        self._optimize_model()

    def _set_variables(self):
        self._set_edge_delta_variables()
        self._set_h_delta_variables()
        self._set_c_delta_variables()

    def _set_constraints(self):
        if self.octet_rule:
            self.model.update()
            self._set_delta_constraints()

        self.model.update()
        self._set_reactant_center_constraints()

    def _add_idx_to_delta_vars(self, idx, delta_var, delta):
        if idx in self.idx_to_delta_vars:
            self.idx_to_delta_vars[idx].append((delta, delta_var))
        else:
            self.idx_to_delta_vars[idx] = [(delta, delta_var)]

    @staticmethod
    def _build_c_delta_var_name(i, c_delta):
        return '{}_{}_{}'.format(C_DELTA_VAR_NAME_HEADER, i, c_delta)

    @staticmethod
    def _build_h_delta_var_name(i, h_delta):
        return '{}_{}_{}'.format(H_DELTA_VAR_NAME_HEADER, i, h_delta)

    @staticmethod
    def _build_edge_delta_var_name(i, j, edge_delta):
        return '{}_{}_{}_{}'.format(EDGE_DELTA_VAR_NAME_HEADER, i, j, edge_delta)

    @staticmethod
    def _parse_var_name(var_name):
        results = var_name.split('_')
        header = results[0]
        results = [int(result) for result in results[1:]]
        return [header] + results

    @staticmethod
    def _get_clip_prob_idx_set(probas):
        sort_by_prob_idxs = np.argsort(probas)[::-1]
        clip_prob_idxs = set()
        cum_prob = 0.0
        for idx in sort_by_prob_idxs:
            clip_prob_idxs.add(idx)
            cum_prob += probas[idx]
            if cum_prob > CLIP_PROB_THRESHOLD:
                break
        return clip_prob_idxs

    def _set_edge_delta_variables(self):

        def _edge_range(bond_idx):
            min_delta = 0
            max_delta = 6
            if bond_idx in self.reactant_bond_map:
                val = BOND_TYPE_TO_CHARGE[self.reactant_bond_map[bond_idx].GetBondType()]
                min_delta -= val
                max_delta -= val
            return min_delta, max_delta

        for i in range(self.n_atom):
            for j in range(i + 1, self.n_atom):
                if i in self.reactant_atom_idx and j in self.reactant_atom_idx:
                    delta_vars = []
                    delta_probas = []
                    probas = self.edge_calibration_fn(self.edge_delta_pred[i, j])
                    min_delta, max_delta = _edge_range((i, j))
                    for idx in range(len(probas)):
                        if EDGE_DELTA_VAL_LIST[idx] > max_delta or EDGE_DELTA_VAL_LIST[idx] < min_delta:
                            probas[idx] = 0
                    probas /= np.sum(probas)
                    clip_prob_idx_set = self._get_clip_prob_idx_set(probas)
                    clip_prob_idx_set = {idx for idx in clip_prob_idx_set if
                                         min_delta <= EDGE_DELTA_VAL_LIST[idx] <= max_delta}
                    starting_idx = np.argmax(probas)
                    for idx, proba in enumerate(probas):
                        if idx not in clip_prob_idx_set:
                            continue
                        delta = EDGE_DELTA_VAL_LIST[idx]
                        var_name = self._build_edge_delta_var_name(i, j, delta)
                        delta_var = self.model.addVar(vtype=GRB.BINARY, name=var_name)
                        delta_var.start = int(idx == starting_idx)
                        delta_vars.append(delta_var)
                        delta_probas.append(proba)
                        self._add_idx_to_delta_vars(i, delta_var, delta)
                        self._add_idx_to_delta_vars(j, delta_var, delta)
                        if delta != 0:
                            self.reaction_center_delta_vars.append((1, delta_var))

                    if len(delta_vars) > 0:
                        self.model.addConstr(quicksum(delta_vars) == 1)
                        self.model_objective.append(
                            LinExpr(np.log(delta_probas) * self.edge_coefficient, delta_vars))

    def _set_h_delta_variables(self):

        def _h_range(idx):
            val = self.reactant_atom_map[idx].GetTotalNumHs() * 2
            return 0 - val, 8 - val

        for i in range(self.n_atom):
            if i in self.reactant_atom_idx:
                delta_vars = []
                delta_probas = []
                probas = self.h_calibration_fn(self.h_delta_pred[i])
                min_delta, max_delta = _h_range(i)
                for idx in range(len(probas)):
                    if H_DELTA_VAL_LIST[idx] > max_delta or H_DELTA_VAL_LIST[idx] < min_delta:
                        probas[idx] = 0
                probas /= np.sum(probas)
                clip_prob_idx_set = self._get_clip_prob_idx_set(probas)
                clip_prob_idx_set = {idx for idx in clip_prob_idx_set if
                                     min_delta <= H_DELTA_VAL_LIST[idx] <= max_delta}
                starting_idx = np.argmax(probas)
                for idx, proba in enumerate(probas):
                    if idx not in clip_prob_idx_set:
                        continue
                    delta = H_DELTA_VAL_LIST[idx]
                    var_name = self._build_h_delta_var_name(i, delta)
                    delta_var = self.model.addVar(vtype=GRB.BINARY, name=var_name)
                    delta_var.start = int(idx == starting_idx)
                    delta_vars.append(delta_var)
                    delta_probas.append(proba)
                    self._add_idx_to_delta_vars(i, delta_var, delta)

                if len(delta_vars) > 0:
                    self.model.addConstr(quicksum(delta_vars) == 1)
                    self.model_objective.append(
                        LinExpr(np.log(delta_probas) * self.h_coefficient, delta_vars))

    def _set_c_delta_variables(self):

        def _c_range():
            return -2, 6

        for i in range(self.n_atom):
            if i in self.reactant_atom_idx:
                delta_vars = []
                delta_probas = []
                probas = self.c_delta_pred[i]
                min_delta, max_delta = _c_range()
                for idx in range(len(probas)):
                    if C_DELTA_VAL_LIST[idx] > max_delta or C_DELTA_VAL_LIST[idx] < min_delta:
                        probas[idx] = 0
                probas /= np.sum(probas)
                clip_prob_idx_set = self._get_clip_prob_idx_set(probas)
                clip_prob_idx_set = {idx for idx in clip_prob_idx_set if
                                     min_delta <= C_DELTA_VAL_LIST[idx] <= max_delta}
                starting_idx = np.argmax(probas)
                for idx, proba in enumerate(probas):
                    if idx not in clip_prob_idx_set:
                        continue
                    delta = C_DELTA_VAL_LIST[idx]
                    var_name = self._build_c_delta_var_name(i, delta)
                    delta_var = self.model.addVar(vtype=GRB.BINARY, name=var_name)
                    delta_var.start = int(idx == starting_idx)
                    delta_vars.append(delta_var)
                    delta_probas.append(proba)
                    self._add_idx_to_delta_vars(i, delta_var, delta)

                if len(delta_vars) > 0:
                    self.model.addConstr(quicksum(delta_vars) == 1)
                    self.model_objective.append(
                        LinExpr(np.log(delta_probas) * self.c_coefficient, delta_vars))

    def _set_delta_constraints(self):
        # Octet Rule Constraints
        for idx, delta_vars in self.idx_to_delta_vars.items():
            if len(delta_vars) > 1:
                atom = self.reactant_atom_map[idx]
                ub, lb = get_atom_octet_charge_change(atom)
                if ub is not None and lb is not None:
                    expr = LinExpr(delta_vars)
                    self.model.addConstr(expr <= ub, name="Octet Rule Atom: %d UB" % (idx + 1))
                    self.model.addConstr(expr >= lb, name="Octet Rule Atom: %d LB" % (idx + 1))

    def _set_reactant_center_constraints(self):
        # Reaction Center Number Constraints
        self.model.addConstr(LinExpr(self.reaction_center_delta_vars) >= 1)
        self.model.addConstr(LinExpr(self.reaction_center_delta_vars) <= 6)

    def _set_model_param(self):
        self.model.update()
        self.model.setParam('MIPGap', 0.0)
        self.model.setParam('MIPFocus', 2)
        self.model.setParam(GRB.Param.PoolSearchMode, 2)
        self.model.setParam(GRB.Param.PoolSolutions, self.num_candidates)

    def _set_model_objective(self):
        self.model.update()
        self.model.setObjective(quicksum(self.model_objective), GRB.MAXIMIZE)

    def _optimize_model(self):
        self.model.update()
        try:
            self.model.optimize()
        except GurobiError as e:
            print('Sampling error for reaction string:\n{}'.format(self.reaction_str))
            raise e

    def _idx_to_atom_idx(self, idx):
        return self.reactant_atom_map[idx].GetIdx()

    def _build_solution_mol(self, solution_dict):

        solution_mol = Chem.rdchem.EditableMol(self.reactant_mol)

        # Modify Edge.
        for bond_idx, delta in sorted(solution_dict[EDGE_DELTA_VAR_NAME_HEADER].items(),
                                      key=operator.itemgetter(1),
                                      reverse=False):
            if abs(delta) > 0:
                idx1, idx2 = bond_idx
                atom_idx1 = self._idx_to_atom_idx(idx1)
                atom_idx2 = self._idx_to_atom_idx(idx2)
                old_bond = self.reactant_mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)

                old_charge = 0
                if old_bond is not None:
                    old_charge += BOND_TYPE_TO_CHARGE[old_bond.GetBondType()]
                    solution_mol.RemoveBond(atom_idx1, atom_idx2)

                new_charge = old_charge + delta
                if new_charge > 0:
                    solution_mol.AddBond(atom_idx1, atom_idx2, CHARGE_TO_BOND_TYPE[new_charge])
                elif new_charge < 0:
                    raise ValueError('New charge cannot be negative: {}'.format(new_charge))

        # Get the modified solution mol.
        solution_mol = solution_mol.GetMol()
        solution_mol_atom_map = {idxfunc(atom): atom for atom in solution_mol.GetAtoms()}

        # Modify H.
        for idx, delta in solution_dict[H_DELTA_VAR_NAME_HEADER].items():
            if abs(delta) > 0:
                atom = solution_mol_atom_map[idx]
                old_h = int(atom.GetTotalNumHs())
                atom.SetNoImplicit(True)
                new_h = int(old_h + (delta / 2))
                atom.SetNumExplicitHs(new_h)

        # Modify Charge.
        for idx, delta in solution_dict[C_DELTA_VAR_NAME_HEADER].items():
            if abs(delta) > 0:
                atom = solution_mol_atom_map[idx]
                new_c = int(atom.GetFormalCharge() - (delta / 2))
                atom.SetFormalCharge(new_c)

        try:
            solution_mol = Chem.Mol(solution_mol)
            Chem.SanitizeMol(solution_mol)
            Chem.Kekulize(solution_mol, clearAromaticFlags=False)
            Chem.SanitizeMol(solution_mol)
            return solution_mol
        except ValueError as ve:
            return None

    def get_solutions(self):

        num_solutions = min(self.num_candidates, self.model.SolCount)

        solutions = []

        for solution_idx in range(num_solutions):

            self.model.setParam(GRB.Param.SolutionNumber, solution_idx)

            solution_dict = {EDGE_DELTA_VAR_NAME_HEADER: {},
                             H_DELTA_VAR_NAME_HEADER: {},
                             C_DELTA_VAR_NAME_HEADER: {}}

            for solution_var in self.model.getVars():
                state = solution_var.Xn
                var_name = solution_var.varName
                if '-delta' in var_name and state > SELECTION_EPSILON and var_name[var_name.rfind('_') + 1:] != '0':
                    solution_var_result = self._parse_var_name(var_name)
                    if solution_var_result[0] == EDGE_DELTA_VAR_NAME_HEADER:
                        idx1 = solution_var_result[1]
                        idx2 = solution_var_result[2]
                        delta = solution_var_result[3]
                        solution_dict[EDGE_DELTA_VAR_NAME_HEADER][(idx1, idx2)] = delta
                    elif solution_var_result[0] == H_DELTA_VAR_NAME_HEADER:
                        idx = solution_var_result[1]
                        delta = solution_var_result[2]
                        solution_dict[H_DELTA_VAR_NAME_HEADER][idx] = delta
                    elif solution_var_result[0] == C_DELTA_VAR_NAME_HEADER:
                        idx = solution_var_result[1]
                        delta = solution_var_result[2]
                        solution_dict[C_DELTA_VAR_NAME_HEADER][idx] = delta
                    else:
                        raise ValueError('Invalid header string: {}'.format(solution_var_result[0]))

            solution_mol = self._build_solution_mol(solution_dict)

            if solution_mol is None:
                continue

            solution = {SAMPLE_SOLUTION_MOL_KEY: solution_mol,
                        SAMPLE_SOLUTION_VAL_KEY: self.model.PoolObjVal}

            solutions.append(solution)

        return solutions


def run_gurobi_sampler(delta_pred,
                       num_candidates=10,
                       verbose=False):
    solutions = GurobiSampler(delta_pred,
                              num_candidates=num_candidates,
                              verbose=verbose).get_solutions()

    return solutions
