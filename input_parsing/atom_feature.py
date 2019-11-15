"""Generate atom features."""

import numpy as np
from rdkit import Chem

from input_parsing.util import one_hot_encoding, ELEMENT_LIST, ATOM_FEATURE_DIM, idxfunc, ATOM_FEATURES_KEY

PT = Chem.GetPeriodicTable()

ATOM_CLASS_TABLE = {}
NOBLE_GAS_ATOMIC_NUM = {2, 10, 18, 36, 54, 86}
OTHER_NON_METAL_ATOMIC_NUM = {1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53}
METALLOID_ATOMIC_NUM = {5, 14, 32, 33, 51, 52, 85}
POST_TRANSITION_METAL_ATOMIC_NUM = {13, 31, 49, 50, 81, 82, 83, 84, 114}
TRANSITION_METAL_ATOMIC_NUM = set(range(21, 30 + 1)) | set(range(39, 48 + 1)) | set(range(72, 80 + 1)) | set(
    range(104, 108 + 1)) | {112}
ALKALI_METAL_ATOMIC_NUM = {3, 11, 19, 37, 55, 87}
ALKALI_EARCH_METAL_ATOMIC_NUM = {4, 12, 20, 38, 56, 88}
LANTHANOID_ATOMIC_NUM = set(range(57, 71 + 1))
ACTINOID_ATOMIC_NUM = set(range(89, 103 + 1))
ATOM_CLASSES = [NOBLE_GAS_ATOMIC_NUM, OTHER_NON_METAL_ATOMIC_NUM, METALLOID_ATOMIC_NUM,
                POST_TRANSITION_METAL_ATOMIC_NUM, TRANSITION_METAL_ATOMIC_NUM,
                ALKALI_EARCH_METAL_ATOMIC_NUM, ALKALI_METAL_ATOMIC_NUM, LANTHANOID_ATOMIC_NUM, ACTINOID_ATOMIC_NUM]
for class_index, atom_class in enumerate(ATOM_CLASSES):
    for num in atom_class:
        ATOM_CLASS_TABLE[num] = class_index + 1

ALLEN_NEGATIVITY_TABLE = {
    1: 2.3,
    2: 4.16,
    3: 0.912,
    4: 1.576,
    5: 2.051,
    6: 2.544,
    7: 3.066,
    8: 3.61,
    9: 4.193,
    10: 4.787,
    11: 0.869,
    12: 1.293,
    13: 1.613,
    14: 1.916,
    15: 2.253,
    16: 2.589,
    17: 2.869,
    18: 3.242,
    19: 0.734,
    20: 1.034,
    21: 1.19,
    22: 1.38,
    23: 1.53,
    24: 1.65,
    25: 1.75,
    26: 1.8,
    27: 1.84,
    28: 1.88,
    29: 1.85,
    30: 1.59,
    31: 1.756,
    32: 1.994,
    33: 2.211,
    34: 2.424,
    35: 2.685,
    36: 2.966,
    37: 0.706,
    38: 0.963,
    39: 1.12,
    40: 1.32,
    41: 1.41,
    42: 1.47,
    43: 1.51,
    44: 1.54,
    45: 1.56,
    46: 1.58,
    47: 1.87,
    48: 1.52,
    49: 1.656,
    50: 1.824,
    51: 1.984,
    52: 2.158,
    53: 2.359,
    54: 2.582,
    55: 0.659,
    56: 0.881,
    71: 1.09,
    72: 1.16,
    73: 1.34,
    74: 1.47,
    75: 1.6,
    76: 1.65,
    77: 1.68,
    78: 1.72,
    79: 1.92,
    80: 1.76,
    81: 1.789,
    82: 1.854,
    83: 2.01,
    84: 2.19,
    85: 2.39,
    86: 2.6,
    87: 0.67,
    88: 0.89
}

ELECTRON_AFFINITY_TABLE = (
    (1, 0.75),
    (1, 0.75),
    (2, -0.52),
    (3, 0.62),
    (4, -0.52),
    (5, 0.28),
    (6, 1.26),
    (6, 1.26),
    (7, 0.00),
    (7, 0.01),
    (7, 0.01),
    (8, 1.46),
    (8, 1.46),
    (8, 1.46),
    (8, -7.71),
    (9, 3.40),
    (10, -1.20),
    (11, 0.55),
    (12, -0.41),
    (13, 0.43),
    (14, 1.39),
    (15, 0.75),
    (15, -4.85),
    (15, -9.18),
    (16, 2.08),
    (16, 2.08),
    (16, -4.72),
    (17, 3.61),
    (18, -1.00),
    (19, 0.50),
    (20, 0.02),
    (21, 0.19),
    (22, 0.08),
    (23, 0.53),
    (24, 0.68),
    (25, -0.52),
    (26, 0.15),
    (27, 0.66),
    (28, 1.16),
    (29, 1.24),
    (30, -0.62),
    (31, 0.43),
    (32, 1.23),
    (33, 0.80),
    (34, 2.02),
    (35, 3.36),
    (36, -0.62),
    (37, 0.49),
    (38, 0.05),
    (39, 0.31),
    (40, 0.43),
    (41, 0.92),
    (42, 0.75),
    (43, 0.55),
    (44, 1.05),
    (45, 1.14),
    (46, 0.56),
    (47, 1.30),
    (48, -0.72),
    (49, 0.30),
    (50, 1.11),
    (51, 1.05),
    (52, 1.97),
    (53, 3.06),
    (54, -0.83),
    (55, 0.47),
    (56, 0.14),
    (57, 0.47),
    (58, 0.65),
    (59, 0.96),
    (60, 1.92),
    (61, 0.13),
    (62, 0.16),
    (63, 0.86),
    (64, 0.14),
    (65, 1.17),
    (66, 0.35),
    (67, 0.34),
    (68, 0.31),
    (69, 1.03),
    (70, -0.02),
    (71, 0.35),
    (72, 0.02),
    (73, 0.32),
    (74, 0.82),
    (75, 0.06),
    (76, 1.10),
    (77, 1.56),
    (78, 2.13),
    (79, 2.31),
    (80, -0.52),
    (81, 0.38),
    (82, 0.36),
    (83, 0.94),
    (84, 1.90),
    (85, 2.30),
    (86, -0.72),
    (87, 0.49),
    (88, 0.10),
    (89, 0.35),
    (90, 1.17),
    (91, 0.55),
    (92, 0.53),
    (93, 0.48),
    (94, -0.50),
    (95, 0.10),
    (96, 0.28),
    (97, -1.72),
    (98, -1.01),
    (99, -0.30),
    (100, 0.35),
    (101, 0.98),
    (102, -2.33),
    (103, -0.31),
    (111, 1.57),
    (113, 0.69),
    (115, 0.37),
    (116, 0.78),
    (117, 1.72),
    (118, 0.06),
    (119, 0.66),
    (120, 0.02),
    (121, 0.57),
)
ELECTRON_AFFINITY_TABLE = {k: v for (k, v) in ELECTRON_AFFINITY_TABLE}


def _get_atomic_features(atomic_num):
    # Symbol
    symbol = PT.GetElementSymbol(atomic_num)
    symbol_k = one_hot_encoding(symbol, ELEMENT_LIST)

    # Period
    outer_electrons = PT.GetNOuterElecs(atomic_num)
    outer_electrons_k = one_hot_encoding(outer_electrons, list(range(0, 8 + 1)))

    # Default Valence
    default_electrons = PT.GetDefaultValence(atomic_num)  # -1 for transition metals
    default_electrons_k = one_hot_encoding(default_electrons, list(range(-1, 8 + 1)))

    # Orbitals / Group / ~Row
    orbitals = next(j + 1 for j, val in enumerate([2, 10, 18, 36, 54, 86, 120]) if val >= atomic_num)
    orbitals_k = one_hot_encoding(orbitals, list(range(0, 7 + 1)))

    # IUPAC Series
    atom_series = ATOM_CLASS_TABLE[atomic_num]
    atom_series_k = one_hot_encoding(atom_series, list(range(0, 9 + 1)))

    # Centered Electrons
    centered_oec = abs(outer_electrons - 4)

    # Electronegativity & Electron Affinity
    try:
        allen_electronegativity = ALLEN_NEGATIVITY_TABLE[atomic_num]
    except KeyError:
        allen_electronegativity = 0
    try:
        electron_affinity = ELECTRON_AFFINITY_TABLE[atomic_num]
    except KeyError:
        electron_affinity = 0

    # Mass & Radius (van der waals / covalent / bohr 0)
    floats = [centered_oec, allen_electronegativity, electron_affinity, PT.GetAtomicWeight(atomic_num),
              PT.GetRb0(atomic_num), PT.GetRvdw(atomic_num), PT.GetRcovalent(atomic_num), outer_electrons,
              default_electrons, orbitals]
    # print(symbol_k + outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats)

    # Compose feature array
    feature_array = np.array(
        symbol_k + outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats,
        dtype=np.float32
    )

    # Cache in dict for future use
    return feature_array


def _get_atom_is_conjugated(atom):
    for bond in atom.GetBonds():
        if bond.GetIsConjugated():
            return True
    return False


def _get_atom_features(atom):
    h = atom.GetTotalNumHs()  # -1 for transition metals
    h_k = one_hot_encoding(h, list(range(5)))

    d = atom.GetDegree()
    d_k = one_hot_encoding(d, list(range(5)))

    ev = atom.GetExplicitValence()
    ev_k = one_hot_encoding(ev, list(range(9)))

    iv = atom.GetImplicitValence()
    iv_k = one_hot_encoding(iv, list(range(9)))

    hyb = atom.GetHybridization()
    hyb_k = one_hot_encoding(str(hyb), ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'other'])

    c = atom.GetFormalCharge()
    arom = atom.GetIsAromatic()
    ring = atom.IsInRing()
    radical_e = atom.GetNumRadicalElectrons()
    conjugated = _get_atom_is_conjugated(atom)

    return np.array(h_k
                    + d_k
                    + ev_k
                    + iv_k
                    + hyb_k
                    + [c, arom, ring, radical_e, conjugated], dtype=np.float)


def get_mol_atom_features(reactant_mol, num_atom=None, reactant_atom_idx=None):
    if num_atom is None:
        num_atom = reactant_mol.GetNumAtoms()
    atom_features = np.zeros((num_atom, ATOM_FEATURE_DIM), dtype=np.float)

    for atom in reactant_mol.GetAtoms():
        idx = idxfunc(atom)
        atom_feature = np.append(_get_atomic_features(atom.GetAtomicNum()), _get_atom_features(atom))
        atom_features[idx, :ATOM_FEATURE_DIM - 1] = atom_feature
        if reactant_atom_idx is not None and idx in reactant_atom_idx:
            atom_features[idx, -1] = 1

    return {ATOM_FEATURES_KEY: atom_features}
