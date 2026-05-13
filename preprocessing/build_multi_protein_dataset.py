import os
import numpy as np
from Bio.PDB import PDBParser

try:
    from Bio.PDB.SASA import ShrakeRupley
except ImportError:
    ShrakeRupley = None


# ----------------------------
# Config
# ----------------------------
CONTACT_THRESHOLD = 5.0
INTRA_CA_EDGE_THRESHOLD = 8.0
CANDIDATE_RADIUS = 12.0
USE_CANDIDATE_FILTER = True
MAX_CORR_EDGES = 3_000_000
PROGRESS_EVERY = 2000

# Feature modes:
#   "basic"                -> [CA_distance, degree_A, degree_B]
#   "aa_onehot"            -> basic + aa_A_onehot(20) + aa_B_onehot(20)
#   "physicochemical"      -> basic + hydrophobicity/charge/polarity/aromaticity for both residues
#   "basic_asa"            -> basic + ASA_A + ASA_B
#   "physicochemical_asa"  -> physicochemical + ASA_A + ASA_B
FEATURE_MODE = "basic"


# ----------------------------
# Amino acid encodings
# ----------------------------
STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AA)}


def amino_acid_one_hot(resname):
    """
    Convert 3-letter amino acid code to one-hot vector.
    Unknown residues are encoded as all zeros.
    """
    vec = np.zeros(len(STANDARD_AA), dtype=np.float32)

    if resname in AA_TO_INDEX:
        vec[AA_TO_INDEX[resname]] = 1.0

    return vec


# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}

# Simple approximate charge at physiological pH
CHARGE = {
    "ARG": 1.0,
    "LYS": 1.0,
    "HIS": 1.0,
    "ASP": -1.0,
    "GLU": -1.0,
}

POLAR_AA = {
    "ARG", "ASN", "ASP", "GLN", "GLU",
    "HIS", "LYS", "SER", "THR", "TYR", "CYS",
}

AROMATIC_AA = {
    "PHE", "TRP", "TYR", "HIS",
}


def physicochemical_features(resname):
    """
    Return compact physicochemical features for a residue.

    Features:
      [hydrophobicity, charge, polarity, aromaticity]

    Unknown residues are encoded as zeros.
    """
    hydrophobicity = HYDROPHOBICITY.get(resname, 0.0)
    charge = CHARGE.get(resname, 0.0)
    polarity = 1.0 if resname in POLAR_AA else 0.0
    aromaticity = 1.0 if resname in AROMATIC_AA else 0.0

    return np.array(
        [hydrophobicity, charge, polarity, aromaticity],
        dtype=np.float32,
    )


# ----------------------------
# Utility functions
# ----------------------------
def load_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", pdb_path)


def compute_residue_asa(model):
    """
    Compute residue-level accessible surface area using Biopython ShrakeRupley.

    The computed residue ASA is attached to each residue as:
      residue.sasa

    If ShrakeRupley is unavailable, ASA values will default to 0.
    """
    if ShrakeRupley is None:
        print("Warning: Bio.PDB.SASA.ShrakeRupley not available. ASA will be set to 0.")
        return

    sr = ShrakeRupley()
    sr.compute(model, level="R")


def extract_partner_residues(model, chain_ids):
    """
    chain_ids can be:
      "A"
      "AB"
      "HL"
      "LH"
      etc.

    All residues from the listed chains are merged into one partner.

    Returns:
      residues_atoms: list of atom coordinates for each residue
      ca_coords: C-alpha coordinates
      residue_names: 3-letter amino acid names
      residue_asa: residue-level accessible surface area values
    """
    residues_atoms = []
    ca_coords = []
    residue_names = []
    residue_asa = []

    for chain_id in chain_ids:
        if chain_id not in model:
            available = list(model.child_dict.keys())
            raise ValueError(
                f"Chain {chain_id} not found. Available chains: {available}"
            )

        chain = model[chain_id]

        for res in chain:
            # Keep only standard protein residues
            if res.get_id()[0] != " ":
                continue

            atoms = [atom.get_coord() for atom in res]

            if not atoms or not res.has_id("CA"):
                continue

            residues_atoms.append(np.array(atoms, dtype=np.float32))
            ca_coords.append(res["CA"].get_coord())
            residue_names.append(res.get_resname())

            asa_value = float(getattr(res, "sasa", 0.0))
            residue_asa.append(asa_value)

    return (
        residues_atoms,
        np.array(ca_coords, dtype=np.float32),
        residue_names,
        np.array(residue_asa, dtype=np.float32),
    )


def build_edge_index_from_ca(ca_coords, threshold):
    n = len(ca_coords)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(ca_coords[i] - ca_coords[j]) < threshold:
                edges.append((i, j))
                edges.append((j, i))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array(edges, dtype=np.int64).T


def build_adjacency_list(edge_index, num_nodes):
    adj = [set() for _ in range(num_nodes)]

    if edge_index.size == 0:
        return adj

    for src, dst in edge_index.T:
        adj[int(src)].add(int(dst))

    return adj


def residues_in_contact(resA_atoms, resB_atoms, threshold):
    for a in resA_atoms:
        for b in resB_atoms:
            if np.linalg.norm(a - b) < threshold:
                return 1
    return 0


def candidate_filter(ca_A, ca_B, radius):
    keep_A = np.array([
        np.any(np.linalg.norm(ca_B - ca_A[i], axis=1) < radius)
        for i in range(len(ca_A))
    ])

    keep_B = np.array([
        np.any(np.linalg.norm(ca_A - ca_B[j], axis=1) < radius)
        for j in range(len(ca_B))
    ])

    return np.where(keep_A)[0], np.where(keep_B)[0]


def build_correspondence_nodes_and_labels(
    resA_atoms_list,
    resB_atoms_list,
    idx_A,
    idx_B,
    threshold
):
    nA = len(idx_A)
    nB = len(idx_B)
    total = nA * nB

    labels = np.zeros(total, dtype=np.int64)
    pairs = np.zeros((total, 2), dtype=np.int64)

    t = 0

    for a_local, a_idx in enumerate(idx_A):
        resA_atoms = resA_atoms_list[a_idx]

        for b_local, b_idx in enumerate(idx_B):
            resB_atoms = resB_atoms_list[b_idx]

            pairs[t] = (a_idx, b_idx)
            labels[t] = residues_in_contact(
                resA_atoms,
                resB_atoms,
                threshold
            )

            t += 1

            if t % PROGRESS_EVERY == 0:
                print(f"Processed {t}/{total} residue pairs...")

    return labels, pairs, nA, nB


def build_correspondence_edges(
    adjA,
    adjB,
    pairs,
    idx_A,
    idx_B,
    nA_local,
    nB_local
):
    edges = []
    total_nodes = nA_local * nB_local

    mapA = {int(g): int(l) for l, g in enumerate(idx_A)}
    mapB = {int(g): int(l) for l, g in enumerate(idx_B)}

    for corr_id in range(total_nodes):
        a_g, b_g = pairs[corr_id]
        a_g = int(a_g)
        b_g = int(b_g)

        for a2_g in adjA[a_g]:
            if a2_g not in mapA:
                continue

            a2_l = mapA[a2_g]

            for b2_g in adjB[b_g]:
                if b2_g not in mapB:
                    continue

                b2_l = mapB[b2_g]
                corr2_id = a2_l * nB_local + b2_l

                edges.append((corr_id, corr2_id))

                if len(edges) >= MAX_CORR_EDGES:
                    print(
                        f"Reached MAX_CORR_EDGES={MAX_CORR_EDGES}. "
                        "Stopping edge generation."
                    )
                    return np.array(edges, dtype=np.int64).T

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    edges_undirected = []

    for u, v in edges:
        edges_undirected.append((u, v))
        edges_undirected.append((v, u))

    return np.array(edges_undirected, dtype=np.int64).T


def build_feature_vector(
    a_idx,
    b_idx,
    caA,
    caB,
    degree_A,
    degree_B,
    resA_names,
    resB_names,
    asaA,
    asaB,
):
    """
    Build node features for one correspondence node.
    """
    ca_dist = np.linalg.norm(caA[a_idx] - caB[b_idx])
    degA = degree_A[a_idx]
    degB = degree_B[b_idx]

    basic_features = np.array(
        [ca_dist, degA, degB],
        dtype=np.float32,
    )

    asa_features = np.array(
        [asaA[a_idx], asaB[b_idx]],
        dtype=np.float32,
    )

    if FEATURE_MODE == "basic":
        return basic_features

    if FEATURE_MODE == "aa_onehot":
        aaA_onehot = amino_acid_one_hot(resA_names[a_idx])
        aaB_onehot = amino_acid_one_hot(resB_names[b_idx])

        return np.concatenate([
            basic_features,
            aaA_onehot,
            aaB_onehot,
        ])

    if FEATURE_MODE == "physicochemical":
        aaA_physchem = physicochemical_features(resA_names[a_idx])
        aaB_physchem = physicochemical_features(resB_names[b_idx])

        return np.concatenate([
            basic_features,
            aaA_physchem,
            aaB_physchem,
        ])

    if FEATURE_MODE == "basic_asa":
        return np.concatenate([
            basic_features,
            asa_features,
        ])

    if FEATURE_MODE == "physicochemical_asa":
        aaA_physchem = physicochemical_features(resA_names[a_idx])
        aaB_physchem = physicochemical_features(resB_names[b_idx])

        return np.concatenate([
            basic_features,
            aaA_physchem,
            aaB_physchem,
            asa_features,
        ])

    raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")


# ----------------------------
# Multi-protein dataset cases
# ----------------------------
# Format:
# "PDB_ID": ("partner_1_chains", "partner_2_chains")
cases = {
    # Original usable examples
    "1BRS": ("A", "B"),
    "1FSS": ("A", "B"),

    # DBD v5 complex cases
    "1AHW": ("AB", "C"),
    "1DQJ": ("AB", "C"),
    "1E6J": ("HL", "P"),
    "1JPS": ("HL", "T"),
    "1MLC": ("AB", "E"),
    "1WEJ": ("HL", "F"),
    "2FD6": ("HL", "U"),
    "2VIS": ("AB", "C"),
    "3HMX": ("LH", "AB"),
    "3MJ9": ("HL", "A"),
}


# ----------------------------
# Main
# ----------------------------
raw_dir = os.path.join("data", "raw_pdb")
out_dir = os.path.join("data", "processed")
os.makedirs(out_dir, exist_ok=True)

summary = []

print(f"Feature mode: {FEATURE_MODE}")

for pdb_id, (partner1_chains, partner2_chains) in cases.items():
    print(f"\nProcessing {pdb_id} ({partner1_chains} vs {partner2_chains}) ...")

    pdb_file = os.path.join(raw_dir, f"{pdb_id}.pdb")

    if not os.path.exists(pdb_file):
        print(f"Warning: {pdb_file} not found. Skipping {pdb_id}.")
        continue

    structure = load_structure(pdb_file)
    model0 = next(structure.get_models())

    # Compute residue-level ASA before extracting residues.
    compute_residue_asa(model0)

    try:
        resA_atoms, caA, resA_names, asaA = extract_partner_residues(
            model0,
            partner1_chains
        )
        resB_atoms, caB, resB_names, asaB = extract_partner_residues(
            model0,
            partner2_chains
        )
    except ValueError as e:
        print(f"Warning: {e}. Skipping {pdb_id}.")
        continue

    print(f"Partner 1 chains: {partner1_chains}, residues: {len(resA_atoms)}")
    print(f"Partner 2 chains: {partner2_chains}, residues: {len(resB_atoms)}")

    if len(resA_atoms) == 0 or len(resB_atoms) == 0:
        print(f"Warning: Empty partner in {pdb_id}. Skipping.")
        continue

    # Build intra-partner graphs
    edge_index_A = build_edge_index_from_ca(caA, INTRA_CA_EDGE_THRESHOLD)
    edge_index_B = build_edge_index_from_ca(caB, INTRA_CA_EDGE_THRESHOLD)

    print(f"Graph A edges: {edge_index_A.shape[1]}")
    print(f"Graph B edges: {edge_index_B.shape[1]}")

    adjA = build_adjacency_list(edge_index_A, len(resA_atoms))
    adjB = build_adjacency_list(edge_index_B, len(resB_atoms))

    degree_A = np.array([len(neighbors) for neighbors in adjA])
    degree_B = np.array([len(neighbors) for neighbors in adjB])

    # Candidate filtering
    if USE_CANDIDATE_FILTER and CANDIDATE_RADIUS is not None:
        idx_A, idx_B = candidate_filter(caA, caB, CANDIDATE_RADIUS)
        print(
            f"Candidate filter ON (radius={CANDIDATE_RADIUS}Å): "
            f"Keep A: {len(idx_A)}, Keep B: {len(idx_B)}"
        )
    else:
        idx_A = np.arange(len(resA_atoms), dtype=np.int64)
        idx_B = np.arange(len(resB_atoms), dtype=np.int64)
        print("Candidate filter OFF: using all residues.")

    if len(idx_A) == 0 or len(idx_B) == 0:
        print(f"Warning: Candidate filter removed all residues for {pdb_id}. Skipping.")
        continue

    # Build correspondence nodes + labels
    labels, pairs, nA_local, nB_local = build_correspondence_nodes_and_labels(
        resA_atoms,
        resB_atoms,
        idx_A,
        idx_B,
        CONTACT_THRESHOLD
    )

    # Build node features
    features = []

    for (a_idx, b_idx) in pairs:
        feature_vector = build_feature_vector(
            a_idx=a_idx,
            b_idx=b_idx,
            caA=caA,
            caB=caB,
            degree_A=degree_A,
            degree_B=degree_B,
            resA_names=resA_names,
            resB_names=resB_names,
            asaA=asaA,
            asaB=asaB,
        )

        features.append(feature_vector)

    features = np.array(features, dtype=np.float32)

    print(f"Feature dimension: {features.shape[1]}")

    positive = int(labels.sum())
    negative = int(len(labels) - positive)

    print(
        f"Total correspondence nodes: {len(labels)}, "
        f"Positive: {positive}, Negative: {negative}"
    )

    # Build correspondence edges
    corr_edge_index = build_correspondence_edges(
        adjA,
        adjB,
        pairs,
        idx_A,
        idx_B,
        nA_local,
        nB_local
    )

    print(f"Correspondence edges: {corr_edge_index.shape[1]}")

    # Save outputs
    case_name = f"{pdb_id}_{partner1_chains}_{partner2_chains}"

    np.save(os.path.join(out_dir, f"{case_name}_corr_labels.npy"), labels)
    np.save(os.path.join(out_dir, f"{case_name}_corr_pairs.npy"), pairs)
    np.save(os.path.join(out_dir, f"{case_name}_corr_edge_index.npy"), corr_edge_index)
    np.save(os.path.join(out_dir, f"{case_name}_corr_features.npy"), features)

    print(f"Saved processed data for {case_name} in {out_dir}")

    summary.append({
        "case": case_name,
        "nodes": len(labels),
        "positive": positive,
        "negative": negative,
        "edges": corr_edge_index.shape[1],
        "feature_dim": features.shape[1],
    })


print("\n================ DATASET SUMMARY ================")

total_nodes = 0
total_pos = 0
total_neg = 0

for item in summary:
    total_nodes += item["nodes"]
    total_pos += item["positive"]
    total_neg += item["negative"]

    ratio = item["positive"] / item["nodes"] if item["nodes"] > 0 else 0

    print(
        f"{item['case']}: "
        f"nodes={item['nodes']}, "
        f"positive={item['positive']}, "
        f"negative={item['negative']}, "
        f"pos_ratio={ratio:.4f}, "
        f"edges={item['edges']}, "
        f"feature_dim={item['feature_dim']}"
    )

print("-------------------------------------------------")
print(f"TOTAL nodes: {total_nodes}")
print(f"TOTAL positive: {total_pos}")
print(f"TOTAL negative: {total_neg}")

if total_nodes > 0:
    print(f"TOTAL positive ratio: {total_pos / total_nodes:.4f}")