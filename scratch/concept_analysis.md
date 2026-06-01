# 🧬 CUB Concept Frequency vs Performance Analysis

**Analyzed Checkpoint**: `checkpoints/vit_base_patch14_dinov2/cub_vit_base_patch14_dinov2_latent0_20260601_0015.pt`

This report summarizes the frequency of positive annotations vs the balanced accuracy of the CBM concept head, sorted from rarest to most frequent to diagnose sparse concepts learning.

## 📉 30 Rarest Concepts (Highly Sparse)

| Index | Concept Name | GT Positives | Frequency (%) | Balanced Acc (%) | TPR (%) | TNR (%) |
|---|---|---|---|---|---|---|
| 142 | has_eye_color_green | 0 | 0.00% | 100.00% | 100.00% | 100.00% |
| 137 | has_eye_color_purple | 2 | 0.07% | 50.00% | 0.00% | 100.00% |
| 271 | has_leg_color_green | 4 | 0.14% | 50.00% | 0.00% | 98.13% |
| 141 | has_eye_color_olive | 5 | 0.17% | 50.00% | 0.00% | 100.00% |
| 61 | has_back_color_purple | 8 | 0.28% | 50.00% | 0.00% | 99.83% |
| 42 | has_underparts_color_purple | 8 | 0.28% | 50.00% | 12.50% | 95.53% |
| 82 | has_upper_tail_color_purple | 9 | 0.31% | 50.00% | 0.00% | 99.93% |
| 108 | has_breast_color_purple | 9 | 0.31% | 50.00% | 0.00% | 97.16% |
| 200 | has_belly_color_purple | 9 | 0.31% | 55.14% | 33.33% | 96.50% |
| 170 | has_under_tail_color_purple | 9 | 0.31% | 50.00% | 100.00% | 39.89% |
| 286 | has_bill_color_green | 9 | 0.31% | 50.00% | 0.00% | 98.51% |
| 176 | has_under_tail_color_pink | 10 | 0.35% | 50.00% | 0.00% | 97.30% |
| 265 | has_leg_color_iridescent | 10 | 0.35% | 50.00% | 0.00% | 99.83% |
| 67 | has_back_color_pink | 11 | 0.38% | 50.00% | 9.09% | 99.41% |
| 114 | has_breast_color_pink | 11 | 0.38% | 50.00% | 27.27% | 90.02% |
| 281 | has_bill_color_purple | 11 | 0.38% | 50.00% | 0.00% | 96.29% |
| 18 | has_wing_color_pink | 11 | 0.38% | 50.00% | 0.00% | 99.38% |
| 12 | has_wing_color_purple | 12 | 0.41% | 50.00% | 75.00% | 57.19% |
| 155 | has_forehead_color_purple | 12 | 0.41% | 50.00% | 8.33% | 93.31% |
| 266 | has_leg_color_purple | 12 | 0.41% | 50.00% | 50.00% | 63.12% |
| 143 | has_eye_color_pink | 13 | 0.45% | 50.00% | 0.00% | 100.00% |
| 129 | has_throat_color_pink | 14 | 0.48% | 50.00% | 28.57% | 98.79% |
| 161 | has_forehead_color_pink | 14 | 0.48% | 50.00% | 0.00% | 99.13% |
| 48 | has_underparts_color_pink | 14 | 0.48% | 50.00% | 35.71% | 69.37% |
| 226 | has_shape_owl-like | 15 | 0.52% | 50.00% | 0.00% | 100.00% |
| 185 | has_nape_color_purple | 16 | 0.55% | 50.00% | 37.50% | 98.23% |
| 123 | has_throat_color_purple | 16 | 0.55% | 50.00% | 0.00% | 100.00% |
| 270 | has_leg_color_olive | 16 | 0.55% | 50.00% | 18.75% | 90.45% |
| 302 | has_crown_color_pink | 17 | 0.59% | 50.00% | 0.00% | 100.00% |
| 296 | has_crown_color_purple | 17 | 0.59% | 50.00% | 76.47% | 27.92% |

## 📈 10 Most Frequent Concepts

| Index | Concept Name | GT Positives | Frequency (%) | Balanced Acc (%) | TPR (%) | TNR (%) |
|---|---|---|---|---|---|---|
| 35 | has_upperparts_color_black | 1161 | 40.08% | 54.26% | 0.09% | 99.94% |
| 6 | has_bill_shape_all-purpose | 1175 | 40.56% | 61.99% | 52.09% | 71.43% |
| 20 | has_wing_color_black | 1237 | 42.70% | 55.72% | 0.00% | 100.00% |
| 289 | has_bill_color_black | 1264 | 43.63% | 56.71% | 0.24% | 99.94% |
| 235 | has_shape_perching-like | 1413 | 48.77% | 61.81% | 0.85% | 99.80% |
| 218 | has_size_small_(5_-_9_in) | 1496 | 51.64% | 58.03% | 0.67% | 99.50% |
| 54 | has_breast_pattern_solid | 1555 | 53.68% | 47.35% | 7.33% | 95.45% |
| 244 | has_belly_pattern_solid | 1621 | 55.95% | 50.12% | 0.00% | 100.00% |
| 151 | has_bill_length_shorter_than_head | 1634 | 56.40% | 66.06% | 79.93% | 50.75% |
| 145 | has_eye_color_black | 2462 | 84.98% | 50.00% | 0.00% | 100.00% |
