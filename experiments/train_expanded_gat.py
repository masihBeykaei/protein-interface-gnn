import os
import json
import csv
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

THRESHOLDS = [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_graph(processed_dir, case_name, feature_dim=None):
    x = np.load(os.path.join(processed_dir, f"{case_name}_corr_features.npy"))
    y = np.load(os.path.join(processed_dir, f"{case_name}_corr_labels.npy"))
    edge_index = np.load(os.path.join(processed_dir, f"{case_name}_corr_edge_index.npy"))
    if feature_dim is not None:
        x = x[:, :feature_dim]
    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(y, dtype=torch.long))
    data.case_name = case_name
    return data


def load_split(processed_dir, cases, feature_dim=None):
    return [load_graph(processed_dir, c, feature_dim) for c in cases]


def normalize(train, val, test):
    all_x = torch.cat([d.x for d in train], dim=0)
    mean = all_x.mean(dim=0); std = all_x.std(dim=0); std[std == 0] = 1.0
    for ds in [train, val, test]:
        for d in ds:
            d.x = (d.x - mean) / std
    return train, val, test


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden=16, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, 2, heads=1, concat=False, dropout=dropout)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gat2(x, edge_index)


def collect(model, loader, device):
    model.eval(); ys = []; ps = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prob = F.softmax(model(batch.x, batch.edge_index), dim=1)[:, 1]
            ys.append(batch.y.cpu().numpy()); ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate(y, p, threshold):
    pred = (p >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0,1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0,1])
    return {"threshold": threshold, "precision_1": precision[1], "recall_1": recall[1], "f1_1": f1[1], "accuracy": accuracy_score(y, pred), "support_0": support[0], "support_1": support[1], "cm": cm}


def best_threshold(y, p):
    best = None
    for t in THRESHOLDS:
        m = evaluate(y, p, t)
        if best is None or m["f1_1"] > best["f1_1"]:
            best = m
    return best


def train(train_loader, val_loader, input_dim, device, args):
    model = GAT(input_dim, args.hidden_channels, args.heads, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None; best_epoch = -1; best_t = 0.5; best_val_f1 = -1; patience = 0
    for epoch in range(args.epochs):
        model.train(); total = 0.0
        for batch in train_loader:
            batch = batch.to(device); opt.zero_grad()
            loss = F.cross_entropy(model(batch.x, batch.edge_index), batch.y)
            loss.backward(); opt.step(); total += loss.item()
        vy, vp = collect(model, val_loader, device); vm = best_threshold(vy, vp)
        if vm["f1_1"] > best_val_f1:
            best_val_f1 = vm["f1_1"]; best_epoch = epoch; best_t = vm["threshold"]; best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
        if epoch % 20 == 0:
            print(f"GAT | Epoch {epoch} | Loss={total/len(train_loader):.4f} | Val_F1_1={vm['f1_1']:.4f} | Best_Val_F1_1={best_val_f1:.4f} | Best_Threshold={best_t:.2f}")
        if patience >= args.patience:
            print(f"Early stopping at epoch {epoch}"); break
    model.load_state_dict(best_state)
    return model, best_epoch, best_t, best_val_f1


def save_outputs(args, splits, input_dim, best_epoch, best_t, best_val_f1, test_metrics):
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    tn, fp, fn, tp = test_metrics["cm"].ravel()
    row = {"best_epoch": best_epoch, "best_threshold": best_t, "best_val_f1_1": best_val_f1, "test_precision_1": test_metrics["precision_1"], "test_recall_1": test_metrics["recall_1"], "test_f1_1": test_metrics["f1_1"], "test_accuracy": test_metrics["accuracy"], "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    fields = list(row.keys())
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow(row)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("# Expanded Balanced GAT Results\n\n")
        f.write(f"Input feature dimension: `{input_dim}`\n\n")
        f.write("## Splits\n\n")
        f.write(f"- Train: {', '.join(splits['train'])}\n")
        f.write(f"- Validation: {', '.join(splits['val'])}\n")
        f.write(f"- Test: {', '.join(splits['test'])}\n\n")
        f.write("## Training Selection\n\n")
        f.write("| Best Epoch | Best Threshold | Best Val F1 1 |\n|------------|----------------|---------------|\n")
        f.write(f"| {best_epoch} | {best_t:.2f} | {best_val_f1:.4f} |\n\n")
        f.write("## Test Metrics\n\n")
        f.write("| Precision 1 | Recall 1 | F1 1 | Accuracy |\n|-------------|----------|------|----------|\n")
        f.write(f"| {test_metrics['precision_1']:.4f} | {test_metrics['recall_1']:.4f} | {test_metrics['f1_1']:.4f} | {test_metrics['accuracy']:.4f} |\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write("| True / Pred | Pred 0 | Pred 1 |\n|-------------|--------|--------|\n")
        f.write(f"| True 0 | {tn} | {fp} |\n| True 1 | {fn} | {tp} |\n\n")
        f.write("Previous strict baseline: `GAT + basic 3 features`, Test F1 1 = `0.2361`.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=os.path.join("data", "processed_expanded"))
    ap.add_argument("--feature_dim", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--hidden_channels", type=int, default=16)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_csv", default=os.path.join("experiments", "expanded_gat_results.csv"))
    ap.add_argument("--output_md", default=os.path.join("experiments", "expanded_gat_results.md"))
    args = ap.parse_args(); set_seed(args.seed)
    split_file = os.path.join(args.processed_dir, "split_cases.json")
    with open(split_file, encoding="utf-8") as f: splits = json.load(f)
    train_ds = load_split(args.processed_dir, splits["train"], args.feature_dim)
    val_ds = load_split(args.processed_dir, splits["val"], args.feature_dim)
    test_ds = load_split(args.processed_dir, splits["test"], args.feature_dim)
    train_ds, val_ds, test_ds = normalize(train_ds, val_ds, test_ds)
    input_dim = train_ds[0].x.shape[1]
    print(f"Expanded GAT | input_dim={input_dim}")
    print("Train:", splits["train"]); print("Val:", splits["val"]); print("Test:", splits["test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", device)
    if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0)) for i, ds in enumerate([train_ds, val_ds, test_ds])]
    model, best_epoch, best_t, best_val_f1 = train(loaders[0], loaders[1], input_dim, device, args)
    ty, tp = collect(model, loaders[2], device); tm = evaluate(ty, tp, best_t)
    print(f"\nFinal Test Metrics: P1={tm['precision_1']:.4f} | R1={tm['recall_1']:.4f} | F1={tm['f1_1']:.4f} | Acc={tm['accuracy']:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:"); print(tm["cm"])
    save_outputs(args, splits, input_dim, best_epoch, best_t, best_val_f1, tm)
    print(f"Saved: {args.output_csv}"); print(f"Saved: {args.output_md}")

if __name__ == "__main__":
    main()
