# Comparison (generated 2026-11-10 17:59)

InfoSeek, authors' P17 protocol, k = 10, n = 496. Accuracy in %. Gap = gold first - gold last.

| Model | Gold first | Gold middle | Gold last | First-last gap | Status |
|---|---|---|---|---|---|
| Paper reproduction (authors' reported) | 45.16 | 32.26 | 28.63 | 16.53 | reported in paper |
| Frozen Qwen | 45.56 | 33.27 | 28.83 | 16.73 | measured (1 run, greedy decoding) |
| Standard LoRA SFT | 54.84 | 55.24 | 53.93 | 0.91 | mean of 2 seeds |
| Balanced LoRA SFT | 54.33 | 55.54 | 54.23 | 0.10 | mean of 2 seeds |
| Isolated reading (frozen, mean log-prob) | 30.85 | 30.85 | 30.85 | 0.00 | measured (order-invariant by construction) |

## Paired bootstrap (B = 10,000, seed 42), percentage points

| Contrast | Estimate | 95% CI | CI excludes 0 |
|---|---|---|---|
| Frozen Qwen: gap (first - last) | +16.73 | [+12.90, +20.77] | yes |
| Standard LoRA SFT: gap (first - last) | +0.91 | [-1.21, +3.02] | no |
| Balanced LoRA SFT: gap (first - last) | +0.10 | [-1.81, +2.02] | no |
| Isolated reading (frozen, mean log-prob): gap (first - last) | +0.00 | [+0.00, +0.00] | no |
| Standard LoRA SFT vs Frozen: change in gap | -15.83 | [-20.16, -11.79] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold first | +9.27 | [+4.84, +13.61] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold middle | +21.98 | [+17.74, +26.21] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold last | +25.10 | [+20.67, +29.44] | yes |
| Balanced LoRA SFT vs Frozen: change in gap | -16.63 | [-20.87, -12.50] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold first | +8.77 | [+4.33, +13.21] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold middle | +22.28 | [+18.15, +26.51] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold last | +25.40 | [+21.07, +29.74] | yes |
| Isolated reading (frozen, mean log-prob) vs Frozen: change in gap | -16.73 | [-20.77, -12.90] | yes |
| Isolated reading (frozen, mean log-prob) vs Frozen: accuracy at gold first | -14.72 | [-18.75, -10.89] | yes |
| Isolated reading (frozen, mean log-prob) vs Frozen: accuracy at gold middle | -2.42 | [-6.05, +1.21] | no |
| Isolated reading (frozen, mean log-prob) vs Frozen: accuracy at gold last | +2.02 | [-1.61, +5.65] | no |
| Standard LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold first | +23.99 | [+19.56, +28.43] | yes |
| Standard LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold middle | +24.40 | [+19.86, +28.93] | yes |
| Standard LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold last | +23.08 | [+18.65, +27.62] | yes |
| Balanced LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold first | +23.49 | [+18.95, +28.02] | yes |
| Balanced LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold middle | +24.70 | [+20.16, +29.23] | yes |
| Balanced LoRA SFT vs Isolated reading (frozen, mean log-prob): accuracy at gold last | +23.39 | [+18.85, +27.82] | yes |
| Balanced LoRA SFT vs Standard LoRA SFT: change in gap | -0.81 | [-3.02, +1.41] | no |
| Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold first | -0.50 | [-2.32, +1.31] | no |
| Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold middle | +0.30 | [-1.61, +2.22] | no |
| Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold last | +0.30 | [-1.61, +2.22] | no |
| [k=20] Frozen Qwen: gap (first - last) | +16.53 | [+12.70, +20.56] | yes |
| [k=20] Standard LoRA SFT: gap (first - last) | +0.91 | [-0.91, +2.82] | no |
| [k=20] Balanced LoRA SFT: gap (first - last) | +0.40 | [-1.61, +2.52] | no |
| [k=20] Standard LoRA SFT vs Frozen: change in gap | -15.63 | [-19.76, -11.49] | yes |
| [k=20] Balanced LoRA SFT vs Frozen: change in gap | -16.13 | [-20.36, -11.90] | yes |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: change in gap | -0.50 | [-2.82, +1.71] | no |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold first | -0.60 | [-2.52, +1.31] | no |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold middle | -1.01 | [-2.82, +0.81] | no |
| [k=20] Balanced LoRA SFT vs Standard LoRA SFT: accuracy at gold last | -0.10 | [-2.02, +1.81] | no |
