# E-VQA transfer (generated 2026-11-10 18:49)

Encyclopedic-VQA, authors' P20 protocol (Qwen part), k = 10, n = 500, canonical one-way matcher. Adapters were trained on InfoSeek only.

| Model | Gold first | Gold middle | Gold last | First-last gap | Status |
|---|---|---|---|---|---|
| Paper P20 Qwen (authors' reported) | 63.20 | 38.60 | 36.80 | 26.40 | reported in paper |
| Frozen Qwen | 63.00 | 39.60 | 35.80 | 27.20 | measured |
| Standard LoRA SFT | 27.20 | 26.80 | 24.00 | 3.20 | mean of 2 seeds |
| Balanced LoRA SFT | 27.50 | 28.10 | 27.60 | -0.10 | mean of 2 seeds |

## Paired bootstrap (B = 10,000, seed 42), percentage points

| Contrast | Estimate | 95% CI | CI excludes 0 |
|---|---|---|---|
| Frozen Qwen: gap (first - last) | +27.20 | [+23.00, +31.40] | yes |
| Standard LoRA SFT: gap (first - last) | +3.20 | [+1.40, +5.10] | yes |
| Balanced LoRA SFT: gap (first - last) | -0.10 | [-2.00, +1.90] | no |
| Standard LoRA SFT vs Frozen: change in gap | -24.00 | [-28.40, -19.70] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold first | -35.80 | [-40.20, -31.40] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold middle | -12.80 | [-16.90, -8.70] | yes |
| Standard LoRA SFT vs Frozen: accuracy at gold last | -11.80 | [-15.90, -7.80] | yes |
| Balanced LoRA SFT vs Frozen: change in gap | -27.30 | [-31.80, -22.90] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold first | -35.50 | [-39.90, -31.10] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold middle | -11.50 | [-15.50, -7.50] | yes |
| Balanced LoRA SFT vs Frozen: accuracy at gold last | -8.20 | [-12.30, -4.20] | yes |
| Balanced vs Standard: change in gap | -3.30 | [-5.30, -1.30] | yes |
| Balanced vs Standard: accuracy at gold first | +0.30 | [-1.30, +1.90] | no |
| Balanced vs Standard: accuracy at gold middle | +1.30 | [-0.40, +3.00] | no |
| Balanced vs Standard: accuracy at gold last | +3.60 | [+2.00, +5.20] | yes |
