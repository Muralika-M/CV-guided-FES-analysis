## Collective Variable-Guided Engineering of the Free-Energy Surface of a Small Peptide

This repository contains analysis scripts, example data, and simulation input files used in the study:

**“[Collective Variable-Guided Engineering of the Free-Energy Surface of a Small Peptide](https://arxiv.org/abs/2602.19906)”**

The code reproduces the HLDA analysis, threshold scanning, and figure generation reported in the manuscript.

---

# Repository Structure


```text
analysis/
    hlda_analysis.py
    threshold_scan.py
    hlda_scatter_grid.py
    residue_correlation_heatmaps.py
    residue_correlation_scatters.py

data/
    example_COLVAR.dat
    example_descriptor_matrix.npy
    Tm_table.csv
    ev/
        per_mutant_hlda_EV_indexed.csv
        hlda_tm_heatmap_indexed.csv
    residue/
        wt_res_imp.csv
    cache/
        threshold_idx_cache/

gromacs_inputs/
    mdp_files/
        emin.mdp
        nvt.mdp
        npt.mdp
        md.mdp

plumed/
    plumed.dat

figures/
    scripts_to_generate_figures.py
```

## Quick run

From this folder:

```bash
python figures/scripts_to_generate_figures.py
```

Generated figures are written to:

- `figures/generated/`

## Notes
- `analysis/hlda_analysis.py` can regenerate EV tables from raw `COLVAR_CV_F` and `COLVAR_CV_UF` data if needed.
