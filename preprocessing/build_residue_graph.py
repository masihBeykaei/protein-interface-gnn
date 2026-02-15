import os
import numpy as np
from Bio.PDB import PDBParser


DISTANCE_THRESHOLD = 8.0  # Angstrom


def load_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    return structure


def extract_chain_residues(structure, chain_id):
    model = structure[0]
    chain = model[chain_id]

    residues = []
    coords = []

    for res in chain:
        if res.has_id("CA"):
            ca_atom = res["CA"]
            residues.append(res)
            coords.append(ca_atom.get_coord())

    return residues, np.array(coords)


def build_edge_index(coords, threshold):
    edge_index = []

    num_nodes = len(coords)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(coords[i] - coords[j])

            if dist < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # undirected graph

    return np.array(edge_index).T  # shape [2, num_edges]


if __name__ == "__main__":
    pdb_file = os.path.join("data", "raw_pdb", "1BRS.pdb")

    structure = load_structure(pdb_file)

    # Chain A
    residues_A, coords_A = extract_chain_residues(structure, "A")
    edge_index_A = build_edge_index(coords_A, DISTANCE_THRESHOLD)

    print("Chain A:")
    print("Nodes:", len(residues_A))
    print("Edges:", edge_index_A.shape[1])
    print("--------")

    # Chain B
    residues_B, coords_B = extract_chain_residues(structure, "B")
    edge_index_B = build_edge_index(coords_B, DISTANCE_THRESHOLD)

    print("Chain B:")
    print("Nodes:", len(residues_B))
    print("Edges:", edge_index_B.shape[1])
