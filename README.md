# Econ_5200_PS3
# ECON 5200 PS3: Independent Review of a Staggered-Rollout Programme Evaluation

An independent review, for a state legislature's budget office, of a workforce agency's evaluation of a county-by-county training programme. The agency's two-way fixed effects (TWFE) estimate reports a positive, significant effect. This review asks whether that number means what the agency says it means.

## Contents

| File | What it is |
|---|---|
| `Econ 5200 PS3.ipynb` | The full analysis, in rubric order |
| `src/did_toolkit.py` | Difference-in-differences module, extending Lab 11's `did_utils.py` |
| `ai-appendix.md` | Full AI transcripts for Phase 4: both prompt versions, raw outputs, and every change made |

## How to run

1. Open `Econ 5200 PS3.ipynb` in Google Colab.
2. Run all cells from the top (Runtime › Run all). The simulated panel uses a fixed seed, so every result reproduces exactly.
3. The module cell writes `src/did_toolkit.py`; the next cell imports it back and checks it against the notebook's results.

## What the notebook does

- **Phase 1:** replicates Card and Krueger's minimum-wage two-by-two (+2.75 FTE), by hand and as a regression interaction.
- **Phase 2:** simulates the agency's setting (40 counties in 8 boards, staggered board-level adoption, an effect
