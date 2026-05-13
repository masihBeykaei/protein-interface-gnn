# Refined GAT Attention Analysis

This refinement post-processes the full GAT attention table and creates more interpretable attention subsets.

The original attention extraction used:

```text
Model: GAT
Feature set: basic 3 features
Attention layer: first GATConv layer
Attention aggregation: mean across heads
```

## Global Attention Statistics

| Metric | Value |
|--------|-------|
| Total attention rows | 427060 |
| Unique edges after deduplication | 215804 |
| Self-loop rows | 4548 |
| Non-self rows | 422512 |
| Global mean attention | 0.010650 |
| Global std attention | 0.006727 |
| Global min attention | 0.000813 |
| Global max attention | 0.344496 |

## Refined Output Files

| File | Description |
|------|-------------|
| `experiments\gat_attention_top_non_self_edges.csv` | Top attention edges excluding self-loops |
| `experiments\gat_attention_top_predicted_positive_edges.csv` | Top attention edges connected to predicted-positive nodes |
| `experiments\gat_attention_top_tp_context_edges.csv` | Top attention edges connected to true-positive nodes |
| `experiments\gat_attention_top_fp_context_edges.csv` | Top attention edges connected to false-positive nodes |
| `experiments\gat_attention_top_fn_context_edges.csv` | Top attention edges connected to false-negative nodes |
| `experiments\gat_attention_error_context_edges.csv` | Top attention edges connected to FP/FN error nodes |
| `experiments\figures\gat_attention_distribution_log.png` | Log-scale distribution of mean attention weights |

## Top Refined Subset Sizes

| Subset | Rows Saved |
|--------|------------|
| Non-self top edges | 100 |
| Predicted-positive top edges | 100 |
| TP-context top edges | 100 |
| FP-context top edges | 100 |
| FN-context top edges | 100 |
| Error-context top edges | 100 |

## Interpretation Notes

- The non-self subset removes self-loops, which dominated the previous top-attention table.
- The predicted-positive subset focuses on edges connected to nodes predicted as interface/contact candidates.
- The TP/FP/FN context files make attention analysis more useful for understanding correct detections and errors.
- GAT attention is normalized locally over incoming neighborhoods, so it should not be interpreted as a global biological importance score.
- High attention can indicate local message-passing importance, but not necessarily true interface relevance.
