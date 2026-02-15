from Bio.PDB import PDBParser
import os


def load_structure(pdb_path):
    """
    Load PDB structure from file
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    return structure


def get_chains(structure):
    """
    Return all chains in first model
    """
    model = structure[0]
    chains = list(model.get_chains())
    return chains


def extract_residues(chain):
    """
    Extract residues that contain CA atom
    Returns list of (residue, CA_coordinates)
    """
    residues = []

    for res in chain:
        if res.has_id("CA"):
            ca_atom = res["CA"]
            coord = ca_atom.get_coord()
            residues.append((res, coord))

    return residues


if __name__ == "__main__":
    pdb_file = os.path.join("data", "raw_pdb", "1BRS.pdb")

    structure = load_structure(pdb_file)
    chains = get_chains(structure)

    # فقط A و B
    target_chains = ["A", "B"]

    for chain in chains:
        if chain.id in target_chains:
            print("Chain ID:", chain.id)

            residues = extract_residues(chain)
            print("Number of residues with CA:", len(residues))
            print("--------")
