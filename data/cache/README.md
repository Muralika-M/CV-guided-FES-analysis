# HLDA Cache

This folder contains the threshold index cache used in the HLDA sweep:

- `threshold_idx_cache/<Mutant>/folded_idx.npz`
- `threshold_idx_cache/<Mutant>/unfolded_idx.npz`

It is used by:

- `analysis/hlda_analysis.py`

Note:
- The raw mutant descriptor files (`COLVAR_CV_F`, `COLVAR_CV_UF`) are not duplicated in this folder.
