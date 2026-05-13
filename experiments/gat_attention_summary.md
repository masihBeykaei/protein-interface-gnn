# GAT Attention Visualization Summary

This analysis extracts first-layer GAT attention weights from the current best strict protocol model.

```text
Model: GAT
Features: basic 3 features
Input dimension: 3
Split: train/validation/test
Attention layer: first GATConv layer
Attention aggregation: mean across heads
```

## Training Selection

- Best epoch: `56`
- Best validation threshold: `0.50`
- Best validation F1 for class 1: `0.2571`

## Test Metrics

| Precision 1 | Recall 1 | F1 1 | Accuracy |
|-------------|----------|------|----------|
| 0.1746 | 0.3642 | 0.2361 | 0.9217 |

## Per-Test-Graph Attention Statistics

| Case | Attention Edges | Mean Attention | Std Attention | Min Attention | Max Attention |
|------|-----------------|----------------|---------------|---------------|---------------|
| 1BRS_A_B | 13673 | 0.016456 | 0.008576 | 0.004833 | 0.084481 |
| 1FSS_A_B | 202397 | 0.009946 | 0.005156 | 0.001231 | 0.092183 |
| 3HMX_LH_AB | 210990 | 0.010948 | 0.007653 | 0.000813 | 0.344496 |

## Top Attention Edges

| Rank | Case | Src Node | Dst Node | Self Loop | Mean Attention | Max Attention | Src True | Dst True | Src P1 | Dst P1 | Src Residue Pair | Dst Residue Pair |
|------|------|----------|----------|-----------|----------------|---------------|----------|----------|--------|--------|------------------|------------------|
| 1 | 3HMX_LH_AB | 124 | 124 | 1 | 0.344496 | 0.636994 | 0 | 0 | 0.0001 | 0.0001 | (27, 194) | (27, 194) |
| 2 | 3HMX_LH_AB | 628 | 628 | 1 | 0.243348 | 0.451933 | 0 | 0 | 0.0001 | 0.0001 | (56, 194) | (56, 194) |
| 3 | 3HMX_LH_AB | 964 | 964 | 1 | 0.236728 | 0.422286 | 0 | 0 | 0.0006 | 0.0006 | (214, 194) | (214, 194) |
| 4 | 3HMX_LH_AB | 40 | 40 | 1 | 0.234497 | 0.484673 | 0 | 0 | 0.0001 | 0.0001 | (0, 194) | (0, 194) |
| 5 | 3HMX_LH_AB | 1930 | 1930 | 1 | 0.234126 | 0.464017 | 0 | 0 | 0.0004 | 0.0004 | (275, 194) | (275, 194) |
| 6 | 3HMX_LH_AB | 82 | 82 | 1 | 0.205271 | 0.343271 | 0 | 0 | 0.0002 | 0.0002 | (1, 194) | (1, 194) |
| 7 | 3HMX_LH_AB | 1006 | 1006 | 1 | 0.199373 | 0.253862 | 0 | 0 | 0.0008 | 0.0008 | (215, 194) | (215, 194) |
| 8 | 3HMX_LH_AB | 460 | 460 | 1 | 0.194928 | 0.338100 | 0 | 0 | 0.0002 | 0.0002 | (52, 194) | (52, 194) |
| 9 | 3HMX_LH_AB | 167 | 124 | 0 | 0.194638 | 0.370338 | 0 | 0 | 0.0027 | 0.0001 | (28, 195) | (27, 194) |
| 10 | 3HMX_LH_AB | 167 | 124 | 0 | 0.194638 | 0.370338 | 0 | 0 | 0.0027 | 0.0001 | (28, 195) | (27, 194) |
| 11 | 3HMX_LH_AB | 965 | 1006 | 0 | 0.175216 | 0.227931 | 0 | 0 | 0.0021 | 0.0008 | (214, 195) | (215, 194) |
| 12 | 3HMX_LH_AB | 965 | 1006 | 0 | 0.175216 | 0.227931 | 0 | 0 | 0.0021 | 0.0008 | (214, 195) | (215, 194) |
| 13 | 3HMX_LH_AB | 586 | 586 | 1 | 0.174255 | 0.310729 | 0 | 0 | 0.0001 | 0.0001 | (55, 194) | (55, 194) |
| 14 | 3HMX_LH_AB | 502 | 502 | 1 | 0.169948 | 0.213856 | 0 | 0 | 0.0001 | 0.0001 | (53, 194) | (53, 194) |
| 15 | 3HMX_LH_AB | 1888 | 1888 | 1 | 0.163306 | 0.212664 | 0 | 0 | 0.0009 | 0.0009 | (274, 194) | (274, 194) |
| 16 | 3HMX_LH_AB | 503 | 628 | 0 | 0.161361 | 0.271709 | 0 | 0 | 0.0008 | 0.0001 | (53, 195) | (56, 194) |
| 17 | 3HMX_LH_AB | 503 | 628 | 0 | 0.161361 | 0.271709 | 0 | 0 | 0.0008 | 0.0001 | (53, 195) | (56, 194) |
| 18 | 3HMX_LH_AB | 2308 | 2308 | 1 | 0.160857 | 0.278580 | 0 | 0 | 0.0254 | 0.0254 | (320, 194) | (320, 194) |
| 19 | 3HMX_LH_AB | 250 | 250 | 1 | 0.159805 | 0.294505 | 0 | 0 | 0.0004 | 0.0004 | (30, 194) | (30, 194) |
| 20 | 3HMX_LH_AB | 503 | 586 | 0 | 0.157079 | 0.268857 | 0 | 0 | 0.0008 | 0.0001 | (53, 195) | (55, 194) |

## Output Files

- Full attention table: `experiments\gat_attention_weights.csv`
- Top attention edges: `experiments\gat_attention_top_edges.csv`
- Attention distribution figure: `experiments\figures\gat_attention_distribution.png`
- Summary: `experiments\gat_attention_summary.md`

## Notes

GAT attention weights are normalized over incoming neighborhoods. Therefore, attention values should be interpreted locally rather than as global importance scores.
