import os
import numpy as np
from Bio.PDB import PDBParser

# ----------------------------
# Config
# ----------------------------
# تماس واقعی residue-residue: اگر هر اتمی از A با هر اتمی از B کمتر از این مقدار باشد
CONTACT_THRESHOLD = 5.0  # Å

# گراف درون‌زنجیره‌ای (A و B): یال اگر فاصله Cα < این مقدار باشد
INTRA_CA_EDGE_THRESHOLD = 8.0  # Å

# برای اینکه correspondence خیلی بزرگ نشود:
# فقط residueهایی را نگه می‌داریم که حداقل یک residue در زنجیره مقابل در شعاع این مقدار دارند
CANDIDATE_RADIUS = 12.0  # Å  (None بگذاری یعنی همه residueها)
USE_CANDIDATE_FILTER = True

# سقف ایمن برای تعداد یال‌های correspondence (برای جلوگیری از انفجار حافظه)
MAX_CORR_EDGES = 3_000_000  # قابل تغییر

# گزارش پیشرفت
PROGRESS_EVERY = 2000


# ----------------------------
# PDB loading & extraction
# ----------------------------
def load_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", pdb_path)


def extract_chain_residues(structure, chain_id: str):
    """
    فقط residue های استاندارد پروتئینی را نگه می‌داریم (hetero/water حذف)
    خروجی:
      residues_atoms: list[np.ndarray]  (N_res, n_atoms_i, 3)
      ca_coords: np.ndarray shape (N_res, 3)
    """
    model = structure[0]
    chain = model[chain_id]

    residues_atoms = []
    ca_coords = []

    for res in chain:
        # فقط amino acid های واقعی
        if res.get_id()[0] != " ":
            continue

        atoms = [atom.get_coord() for atom in res]
        if not atoms:
            continue

        # CA لازم داریم برای گراف درون‌زنجیره‌ای و فیلتر کاندید
        if not res.has_id("CA"):
            continue

        residues_atoms.append(np.array(atoms, dtype=np.float32))
        ca_coords.append(res["CA"].get_coord())

    return residues_atoms, np.array(ca_coords, dtype=np.float32)


# ----------------------------
# Graph building utilities
# ----------------------------
def build_edge_index_from_ca(ca_coords: np.ndarray, threshold: float):
    """
    گراف residue-level درون‌زنجیره‌ای بر اساس فاصله Cα
    خروجی edge_index به شکل np.ndarray با سایز (2, E)
    """
    n = len(ca_coords)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if d < threshold:
                edges.append((i, j))
                edges.append((j, i))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array(edges, dtype=np.int64).T


def build_adjacency_list(edge_index: np.ndarray, num_nodes: int):
    """
    edge_index شکل (2, E)
    خروجی: list[set]
    """
    adj = [set() for _ in range(num_nodes)]
    if edge_index.size == 0:
        return adj
    for src, dst in edge_index.T:
        adj[int(src)].add(int(dst))
    return adj


def residues_in_contact(resA_atoms: np.ndarray, resB_atoms: np.ndarray, threshold: float) -> int:
    """
    تماس residue-residue: اگر هر اتمی در A با هر اتمی در B کمتر از threshold باشد.
    برای سادگی brute-force است (برای این ابعاد مشکلی ندارد).
    """
    for a in resA_atoms:
        for b in resB_atoms:
            if np.linalg.norm(a - b) < threshold:
                return 1
    return 0


def candidate_filter(ca_A: np.ndarray, ca_B: np.ndarray, radius: float):
    """
    residueهای A و B را فیلتر می‌کند تا فقط آن‌هایی بمانند که در شعاع radius از زنجیره مقابل هستند.
    خروجی:
      idx_A_keep, idx_B_keep
    """
    nA, nB = len(ca_A), len(ca_B)
    keep_A = np.zeros(nA, dtype=bool)
    keep_B = np.zeros(nB, dtype=bool)

    # برای هر residue در A، اگر به هر residue در B نزدیک بود نگه دار
    for i in range(nA):
        # فاصله تا همه B
        d = np.linalg.norm(ca_B - ca_A[i], axis=1)
        if np.any(d < radius):
            keep_A[i] = True

    for j in range(nB):
        d = np.linalg.norm(ca_A - ca_B[j], axis=1)
        if np.any(d < radius):
            keep_B[j] = True

    idx_A = np.where(keep_A)[0]
    idx_B = np.where(keep_B)[0]
    return idx_A, idx_B


def build_correspondence_nodes_and_labels(resA_atoms_list, resB_atoms_list, idx_A, idx_B, threshold: float):
    """
    نودهای correspondence را فقط برای idx_A × idx_B می‌سازد.
    نگاشت:
      corr_id = a_local * nB_local + b_local
    خروجی:
      labels: np.ndarray (N_corr,)
      pairs:  np.ndarray (N_corr, 2)  شامل اندیس residue اصلی (global) در A و B
    """
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
            labels[t] = residues_in_contact(resA_atoms, resB_atoms, threshold)

            t += 1
            if t % PROGRESS_EVERY == 0:
                print(f"Processed {t}/{total} residue pairs...")

    return labels, pairs, nA, nB


def build_correspondence_edges(adjA, adjB, pairs, idx_A, idx_B, nA_local, nB_local):
    """
    یال‌های correspondence:
    بین (a,b) و (a',b') اگر a~a' در A و b~b' در B
    اینجا adjA و adjB روی گراف‌های اصلی‌اند (global index).
    pairs هم mapping corr_id -> (a_global, b_global)
    """
    edges = []
    total_nodes = nA_local * nB_local

    # برای اینکه سریع‌تر بشه، یک map از global index به local index می‌سازیم
    mapA = {int(g): int(l) for l, g in enumerate(idx_A)}
    mapB = {int(g): int(l) for l, g in enumerate(idx_B)}

    # برای هر نود correspondence
    for corr_id in range(total_nodes):
        a_g, b_g = pairs[corr_id]
        a_g = int(a_g)
        b_g = int(b_g)

        # همسایه‌های a در A (global)
        for a2_g in adjA[a_g]:
            if a2_g not in mapA:
                continue
            a2_l = mapA[a2_g]

            # همسایه‌های b در B (global)
            for b2_g in adjB[b_g]:
                if b2_g not in mapB:
                    continue
                b2_l = mapB[b2_g]

                corr2_id = a2_l * nB_local + b2_l

                edges.append((corr_id, corr2_id))

                if len(edges) >= MAX_CORR_EDGES:
                    print(f"Reached MAX_CORR_EDGES={MAX_CORR_EDGES}. Stopping edge generation.")
                    return np.array(edges, dtype=np.int64).T

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    # چون گراف بدون جهت می‌خوایم، هر یال را دوطرفه می‌کنیم
    edges_undirected = []
    for u, v in edges:
        edges_undirected.append((u, v))
        edges_undirected.append((v, u))

    return np.array(edges_undirected, dtype=np.int64).T


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    pdb_file = os.path.join("data", "raw_pdb", "1BRS.pdb")

    structure = load_structure(pdb_file)

    # Extract residues + CA coords (global indices)
    resA_atoms, caA = extract_chain_residues(structure, "A")
    resB_atoms, caB = extract_chain_residues(structure, "B")

    print("Residues in Chain A:", len(resA_atoms))
    print("Residues in Chain B:", len(resB_atoms))

    # Build intra-chain graphs
    edge_index_A = build_edge_index_from_ca(caA, INTRA_CA_EDGE_THRESHOLD)
    edge_index_B = build_edge_index_from_ca(caB, INTRA_CA_EDGE_THRESHOLD)

    print("Graph A edges:", edge_index_A.shape[1])
    print("Graph B edges:", edge_index_B.shape[1])

    adjA = build_adjacency_list(edge_index_A, len(resA_atoms))
    adjB = build_adjacency_list(edge_index_B, len(resB_atoms))

    degree_A = np.array([len(neighbors) for neighbors in adjA])
    degree_B = np.array([len(neighbors) for neighbors in adjB])

    # Candidate filtering (optional but recommended)
    if USE_CANDIDATE_FILTER and CANDIDATE_RADIUS is not None:
        idx_A, idx_B = candidate_filter(caA, caB, CANDIDATE_RADIUS)
        print(f"Candidate filter ON (radius={CANDIDATE_RADIUS}Å):")
        print("  Keep A residues:", len(idx_A), "/", len(resA_atoms))
        print("  Keep B residues:", len(idx_B), "/", len(resB_atoms))
    else:
        idx_A = np.arange(len(resA_atoms), dtype=np.int64)
        idx_B = np.arange(len(resB_atoms), dtype=np.int64)
        print("Candidate filter OFF: using all residues.")

    print("Building correspondence nodes + labels...")
    labels, pairs, nA_local, nB_local = build_correspondence_nodes_and_labels(
        resA_atoms, resB_atoms, idx_A, idx_B, CONTACT_THRESHOLD
    )
    # Build node features
    features = []

    for (a_idx, b_idx) in pairs:
        ca_dist = np.linalg.norm(caA[a_idx] - caB[b_idx])
        degA = degree_A[a_idx]
        degB = degree_B[b_idx]

        features.append([ca_dist, degA, degB])

    features = np.array(features, dtype=np.float32)


    print("\n=== Node/Label Results ===")
    print("Total correspondence nodes:", len(labels))
    print("Positive contacts:", int(labels.sum()))
    print("Negative contacts:", int(len(labels) - labels.sum()))

    print("\nBuilding correspondence edges...")
    corr_edge_index = build_correspondence_edges(
        adjA, adjB, pairs, idx_A, idx_B, nA_local, nB_local
    )

    print("\n=== Edge Results ===")
    print("Correspondence edges:", corr_edge_index.shape[1])

    

    # Optional: save intermediate outputs (numpy)
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "corr_labels.npy"), labels)
    np.save(os.path.join(out_dir, "corr_pairs.npy"), pairs)
    np.save(os.path.join(out_dir, "corr_edge_index.npy"), corr_edge_index)
    np.save(os.path.join(out_dir, "corr_features.npy"), features)

    print(f"\nSaved to {out_dir}: corr_labels.npy, corr_pairs.npy, corr_edge_index.npy, corr_features.npy")
