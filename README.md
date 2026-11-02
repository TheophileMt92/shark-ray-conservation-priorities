# CAPTAIN_elasmo_manuscript

Data and code for the manuscript:

> Mouton, T. L., Silvestro, D., Kocáková, K., Spitznagel, D., Leprieur, F., & Pimiento, C. Global gaps and priorities for shark and ray conservation: Integrating threat, function, and evolutionary distinctiveness. *Science Advances* (in revision). Preprint: [bioRxiv 10.1101/2025.11.28.691085](https://www.biorxiv.org/content/10.1101/2025.11.28.691085v1)

Archived version: [Zenodo, doi:10.5281/zenodo.20719267](https://doi.org/10.5281/zenodo.20719267)

The repository contains the inputs, code and outputs needed to reproduce the three components of the study:

1. **Gap analysis**: coverage of the ranges of 1,000 elasmobranch species by no-take MPAs (IUCN categories Ia to III), compared with a spatially explicit null model of MPA placement.
2. **Conservation prioritisation**: spatial priorities for a 10% no-take target, optimised with CAPTAIN for three dimensions of conservation value (IUCN status, FUSE and EDGE2).
3. **Overlap with fishing effort**: post hoc overlap between priorities and industrial fishing effort estimated from AIS and Sentinel-1 SAR vessel detections.

All R analyses are provided as reproducible Quarto reports (`.qmd`), with their rendered HTML versions.

---

## Repository structure

```
.
├── Manuscript repo.Rproj         RStudio project (paths are built with here::here())
├── R/
│   ├── Code/                     Quarto reports (.qmd) and rendered reports (.html)
│   ├── Data/                     Input data and CAPTAIN outputs used by the reports
│   ├── Quarto_template/          Report styling (CSS, header, footer) used by all reports
│   └── Outputs/                  Figures and tables produced by the reports
└── Python/
    ├── captain_script_and_data.zip   CAPTAIN code, training configurations and training logs
    └── *.py                          Scripts summarising CAPTAIN prediction outputs
```

---

## Workflow

Run the steps in the order below. Open `Manuscript repo.Rproj` first so that all `here::here()` paths resolve from the repository root.

### Step 1. Fishing effort estimates (`R/Code/Manuscript_fishing_hours.qmd`)

Combines AIS-based fishing hours from Global Fishing Watch (2017–2020) with Sentinel-1 SAR vessel detections (Paolo et al. 2024). SAR detections are normalised by the number of satellite overpasses per cell, and a random forest model is trained on cells with both data sources to predict fishing hours in cells with SAR detections only.

- **Main output:** `R/Data/Predicted_Fishing_Hours_05Deg_normalised.Rdata` (fishing hours aggregated to the 0.5° analysis grid)
- **Figures and tables:** Figs. S12 to S15 and Tables S5 to S8 (Supplementary Materials)
- **Supporting report:** `SAR overpass normalization - exploratory analysis.qmd` documents the normalisation of SAR detections by satellite overpass frequency.

The raw AIS and SAR files are not included because of their size. They can be downloaded from the [Global Fishing Watch data portal](https://globalfishingwatch.org/data-download/): fishing effort v2 at 0.1° resolution (2017–2020), and SAR vessel detections (Paolo et al. 2024). The overpass counts and distance-to-port and distance-to-shore rasters were obtained from the same source. Steps 2 to 4 only require the processed output listed above.

### Step 2. Gap analysis (`R/Code/Manuscript_gap_analysis.qmd`)

- **Species-level coverage:** the percentage of each species' range covered by no-take MPAs, i.e. range cells inside no-take MPAs divided by all range cells.
- **Coverage by IUCN status:** Jonckheere-Terpstra test, Spearman correlation, Kolmogorov-Smirnov test and quantile regression.
- **Null model:** MPAs randomly placed within each country's EEZ while keeping each country's total protected area (100 iterations). Standardised effect sizes per species, aggregated per grid cell and ecoregion.
- **Main outputs:** `species_mpa_coverage_NT.csv`, under-represented species tables, and ecoregion and EEZ summaries
- **Figures and tables:** Fig. 2 (`Fig2_gap_analysis.png`), Figs. S1 to S8 and Tables S1 and S2

### Step 3. CAPTAIN prioritisation (`Python/captain_script_and_data.zip`)

The zip archive contains the CAPTAIN code (version 2.3), the training script (`train_n_predict_shark.py`), one configuration file per dimension of conservation value (`train_configs/shark_config_iucn.txt`, `shark_config_fuse.txt`, `shark_config_edge.txt`) and the training logs. Key settings, identical across the three models:

| Setting | Value |
|---|---|
| Protection target | 10% of continental grid cells (`protection_target = 0.1`) |
| Existing MPAs | Not used as a starting point (`add_to_existing_protected_areas = False`) |
| Costs | Not used (`use_cost = False`) |
| Reward weights for classes 1 to 5 | 1, 8, 16, 32, 64 (`risk_weights = -64 -32 -16 -8 -1`, listed from class 5 to class 1) |

Species classes are read from `R/Data/continental_shark_conservation_metrics_10_harmonised_IUCN_categories.csv`:

- **IUCN status:** 1 = Least Concern to 5 = Critically Endangered.
- **FUSE and EDGE2:** scores rescaled to a maximum of 1 and binned into five equal-width classes (breaks at 0.2, 0.4, 0.6 and 0.8).

Species ranges are provided as one raster per species in `R/Data/tif files continental/` (1,000 species, 0.5° resolution).

Prediction outputs (`.npz` files, 50 predictions per model) are summarised with the scripts in `Python/`:

- `Analyse CAPTAIN 2 outputs IUCN 220725.py`, `... FUSE 220725.py`, `... EDGE2 220725.py`: cell-level priority (frequency of selection across predictions), saved as `R/Data/CAPTAIN2_*_full_results_averaged_budget0.1_replicates50.rds`
- `Extract protect fraction CAPTAIN 2 0.1 budget_2nd run.py`: protected fraction of each species' range, saved as `R/Data/CAPTAIN2_protected_range_fractions_2ndrun.rds`

Sensitivity to the protection target: `captain_budget_sensitivity_runs.zip` contains CAPTAIN predictions at targets of 5%, 20% and 30% (models trained at 10%, 50 predictions per target and dimension). `Summarise CAPTAIN 2 budget sensitivity runs.py` reads them directly from the zip (no need to unzip) and saves the priority of each cell to `R/Data/CAPTAIN2_budget_sensitivity_priorities.csv` and the protected fraction of each species' range (species in alphabetical order, as in the CAPTAIN input) to `R/Data/CAPTAIN2_budget_sensitivity_protected_range.csv`.

The summarised outputs are included in `R/Data/`, so Step 4 can be run without re-running CAPTAIN. To re-run the Python scripts, edit `dir_path` at the top of each script to point to the local CAPTAIN prediction outputs.

### Step 4. Prioritisation results and overlap with fishing effort (`R/Code/Manuscript_CAPTAIN_analyses.qmd`)

- **Priority maps:** maps for each dimension and pairwise differences between dimensions.
- **Validation:** beta regressions of species' range protection against their IUCN, FUSE and EDGE2 classes.
- **Spatial pattern:** mean nearest-neighbour distances of high-priority cells.
- **Congruence:** congruence of high-priority areas (priority > 0.9) among dimensions, with randomisation tests.
- **Sensitivity to the protection target:** high-priority areas, species protection, nestedness and congruence at targets of 5%, 10%, 20% and 30% (Fig. S16 and Table S9).
- **Overlap with fishing effort:** bivariate maps of priority and fishing effort, and ecoregion-level summaries.
- **Figures:** Fig. 3 (`Fig3_protection_by_class.png`), Fig. 4 (`Fig4_priority_maps.png`), Fig. 5 (`Fig5_priorities_fishing_bivariate.png`), Figs. S9 and S10 (`FigS9_congruence_all_dimensions.png`, `FigS10_congruence_categories.png`), Tables S3 and S4, Fig. S11 (`FigS11_ecoregion_priority_fishing.png`)

Output files in `R/Outputs/` are named after the manuscript figure or table they correspond to (e.g. `Fig4_priority_maps.png`, `FigS10_congruence_categories.png`, `TableS4_congruence_randomisation.png`). Figures shown only in the reports start with `report_`, and output data tables with `data_`.

---

## Input data and sources

| File (in `R/Data/`) | Content | Source |
|---|---|---|
| `tif files continental/*.tif` | Range maps of 1,000 elasmobranch species on a 0.5° grid | IUCN Red List range maps (accessed October 2021), as compiled in Pimiento et al. (2023) |
| `puvsp_marine.Rdata` | Species presence by grid cell | Derived from the range maps |
| `sharks_iucn_final.rds` | IUCN Red List status per species | Pimiento et al. (2023) |
| `continental_shark_conservation_metrics_10_harmonised_IUCN_categories.csv` | IUCN, FUSE and EDGE2 classes used in CAPTAIN | FUSE and EDGE2 scores from Pimiento et al. (2023) |
| `mpa_NT_binary.tif` | No-take MPAs (IUCN categories Ia, Ib, II, III) on the 0.5° grid; a cell is protected if its centre falls within an MPA | World Database on Protected Areas, September 2024 |
| `Predicted_Fishing_Hours_05Deg_normalised.Rdata` | Predicted fishing hours, 2017–2020 | Step 1 |
| `bathymetry-0.1deg-adjusted.tif` | Bathymetry | Global Fishing Watch |
| `meow_ecos/` | Marine Ecoregions of the World | Spalding et al. (2007) |
| `World_High_Seas_v2_20241010/` | High seas boundaries | Marine Regions |
| `World_EEZ_v12_20231025/` (not included) | Exclusive Economic Zones, v12 | [Marine Regions](https://www.marineregions.org/downloads.php); download and place in `R/Data/` |
| `CAPTAIN2_*.rds` | Summarised CAPTAIN outputs | Step 3 |

---

## Software

- **R (version 4.4.2) and Quarto.** Main packages: `sf`, `terra`, `raster`, `tidyverse`, `here`, `betareg`, `quantreg`, `DescTools`, `randomForest`, `caret`, `spatstat`, `biscale`, `patchwork`, `cowplot`, `ggplot2`, `kableExtra`.
- **Python 3 with CAPTAIN 2.3.** See the configuration files in the CAPTAIN archive. Summary scripts use `numpy`, `pandas` and `pyreadr`.

---


## Citation and contact

Please cite the article (or the preprint until publication) and the Zenodo archive when reusing these data or code.

Contact: Théophile L. Mouton (theophile.mouton92@gmail.com)
