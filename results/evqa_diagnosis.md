# E-VQA diagnosis

Saved predictions only; scores unchanged. Canonical = authors' one-way matcher.

| Model | Position | Canonical acc | Any-alternative acc | Mean pred words | Mean gen tokens | Wrong but pred is substring of gold | Exact match |
|---|---|---|---|---|---|---|---|
| frozen | first | 63.0 | 63.0 | 2.4 | 5.3 | 5.4 | 43.2 |
| frozen | middle | 39.6 | 39.6 | 2.5 | 5.5 | 3.0 | 26.6 |
| frozen | last | 35.8 | 35.8 | 2.5 | 5.4 | 2.6 | 24.2 |
| balanced_s0_seed0 | first | 28.0 | 28.0 | 1.5 | 3.8 | 12.2 | 22.2 |
| balanced_s0_seed0 | middle | 29.4 | 29.4 | 1.5 | 3.8 | 11.2 | 23.2 |
| balanced_s0_seed0 | last | 27.8 | 27.8 | 1.5 | 3.8 | 10.8 | 21.0 |
| balanced_s1_seed1 | first | 27.0 | 27.0 | 1.5 | 3.9 | 11.8 | 22.0 |
| balanced_s1_seed1 | middle | 26.8 | 26.8 | 1.5 | 3.8 | 10.6 | 22.2 |
| balanced_s1_seed1 | last | 27.4 | 27.4 | 1.5 | 3.9 | 9.2 | 21.4 |
| standard_seed0 | first | 28.4 | 28.4 | 1.4 | 3.6 | 12.2 | 23.4 |
| standard_seed0 | middle | 27.4 | 27.4 | 1.4 | 3.7 | 12.0 | 22.4 |
| standard_seed0 | last | 24.4 | 24.4 | 1.4 | 3.7 | 11.4 | 20.0 |
| standard_seed1 | first | 26.0 | 26.0 | 1.5 | 3.9 | 11.2 | 20.8 |
| standard_seed1 | middle | 26.2 | 26.2 | 1.5 | 3.9 | 10.6 | 21.4 |
| standard_seed1 | last | 23.6 | 23.6 | 1.4 | 3.9 | 10.4 | 18.6 |

## Canonical accuracy by gold-answer length (gold first)

| Gold words | n | frozen | balanced_s0_seed0 | balanced_s1_seed1 | standard_seed0 | standard_seed1 |
|---|---|---|---|---|---|---|
| 1 | 282 | 71.6 | 38.3 | 36.9 | 41.1 | 35.8 |
| 2-3 | 162 | 59.9 | 19.8 | 18.5 | 16.0 | 17.9 |
| 4-7 | 42 | 35.7 | 0.0 | 2.4 | 0.0 | 0.0 |
| 8+ | 14 | 7.1 | 0.0 | 0.0 | 0.0 | 0.0 |

## Examples: frozen right, standard_seed0 wrong (gold first)

| Question | Gold | Frozen prediction | Standard s0 prediction |
|---|---|---|---|
| What is the typical size of this fungi? |  5–65 cm | 5–65 cm (2.0–25.6 in) | 25.6 |
| What is the wingspan of this insect in mm? | 86 to 89 mm | 86 to 89 mm | 86 |
| What is the habitat of this reptile?  | rocky outcrops | Rocky outcrops on Table Mountain, South Africa | shrubland |
| What is the habitat of this reptile?  | forest and rocky areas. | Forest and rocky areas | forest |
| What does this animal eat? | mussels and barnacles | Mussels and barnacles | algae |
| If this species has a web, what are the features of this web? | no | No web-building behavior | spider web |
| How big can this plant become? |  15–70 cm | 15–70 cm | 60 |
| How big can this plant become? | three meters | Approaching three meters | 60 |
| What does this insect eat? | arboreal lichens | arboreal lichens | lichen |
| In which part of the world does this animal live? | north america | West coast of North America | Brazil |
