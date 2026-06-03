# PD-BBB: Brain, Blood, and CSF Vascular Biomarkers in Parkinson's Disease

Analysis code for the pre-registered multi-modal study testing whether brain-derived, plasma, and cerebrospinal-fluid (CSF) signals of blood-brain barrier (BBB) biology share variance and jointly predict clinical progression in Parkinson's disease (PD).

**Paper:** *Brain, blood, and cerebrospinal fluid vascular biomarkers reflect distinct biology in Parkinson's disease.* Atif Z., Akber U., Kim H.-I., Kwon H.-S., Park C.-S. (2026). Submitted to *npj Parkinson's Disease*.

---

## Repository structure

```
.
├── README.md                       This file
├── environment.yml                 Conda environment specification
├── config.yaml                     Data paths and parameter constants
├── lib/                            Shared Python utilities
│   ├── __init__.py
│   ├── config.py                   Config loader
│   └── ppmi_ids.py                 PPMI ID handling
├── 01_foundation/                  PPMI cohort assembly, clinical canon, V1 baseline reproduction
├── 02_brain/                       snRNA-seq integration, vascular subsetting, DESeq2, WGCNA
├── 03_olink_plasma/                Plasma Olink Explore 384 BBB panel + endotyping
├── 04_strengthening/               Cross-modal decoupling, sensitivity lmer, subgroup fix, hub-gene annotation
├── 05_primary_endpoint/            Pre-registered primary lmer + secondary endpoints + subgroups
├── 06_csf_qalb/                    CSF/serum albumin ratio (Qalb) pipeline
├── 07_csf_olink/                   CSF Olink Explore HT (Project 277) integration, 4-way decoupling
├── figures/                        Figure-rendering scripts (Figs 1-6 + supplementary)
└── manuscript/                     Supplementary table assembly
```

---

## Software requirements

### Python environment

All Python analyses run inside the conda environment specified in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate pd-bbb
```

Core dependencies: Python 3.10, pandas, numpy, scipy, scikit-learn, scanpy, scvi-tools, statsmodels, matplotlib, seaborn, pyarrow, openpyxl, gprofiler-official.

### R environment

A subset of analyses (DESeq2, WGCNA, linear mixed-effects models) run in R. Required packages:

```r
install.packages(c("BiocManager", "lme4", "lmerTest", "WGCNA", "arrow", "tidyverse"))
BiocManager::install(c("DESeq2", "ConsensusClusterPlus"))
```

R 4.2 or later recommended.

---

## Data access

Raw data are **not** included in this repository. All datasets are publicly available, subject to their respective data-use agreements:

### PPMI (Parkinson's Progression Markers Initiative)

- **Source:** https://www.ppmi-info.org/access-data-specimens/download-data
- **Required:** Approved PPMI Data Use Agreement (tier-1 access)
- **Resources used:**
  - Whole-blood RNA-seq (PPMI RNAseq IR3 release)
  - Olink Explore 384 plasma proteomics (Project 9000)
  - Olink Explore HT CSF proteomics (Project 277)
  - CSF and serum albumin (Project 181)
  - Clinical longitudinal data (MDS-UPDRS, MoCA, DaT-SCAN, demographics, biospecimen, genotype calls)

### Brain single-nucleus RNA-seq cohorts (GEO accessions)

| Accession | Citation | Use in this study |
|---|---|---|
| GSE178265 | Kamath et al. (2022) *Nat. Neurosci.* | Primary endothelial atlas (14,903 cells, 18 donors) |
| GSE157783 | Smajic et al. (2022) *Brain* | Smajic midbrain cohort (2,952 vascular cells, 11 donors) |
| GSE140231 | Agarwal et al. (2020) *Nat. Commun.* | Agarwal substantia nigra atlas (265 vascular cells) |

Edit local file paths in `config.yaml` to point to your data locations before running.

---

## Run order

The pipeline is structured as seven sequential stages. Each subfolder is numbered to indicate execution order.

| Stage | What it does | Key outputs |
|---|---|---|
| `01_foundation/` | Build canonical clinical tables; demographics; cohort definitions; baseline V1 reproduction | `results/v1_baseline/`, foundation parquet files |
| `02_brain/` | Load snRNA-seq cohorts; vascular subset; pseudobulk DESeq2; bootstrap WGCNA; module derivation | `results/step2/bbb_module_genes.tsv`, `integrated_de_*.tsv` |
| `03_olink_plasma/` | Load Olink Explore 384 plasma; consensus clustering (k=2); PC1 BBB score | `results/step3/bbb_protein_score.parquet`, endotypes |
| `04_strengthening/` | Cross-modal decoupling (brain vs plasma); sensitivity lmer; corrected genotype-call parsing; hub-gene symbol annotation | `results/step4/cross_modal_scores.parquet`, decoupling stats |
| `05_primary_endpoint/` | Assemble modeling cohort (n=61); pre-registered lmer for UPDRS3; secondaries (MoCA, DaT-SCAN); subgroups | `results/step3/lmer_primary_*.tsv` |
| `06_csf_qalb/` | Compute CSF/serum albumin ratio (Qalb); cross-sectional + longitudinal analyses; 3-way decoupling | `results/step6/qalb_*.parquet`, `.tsv` |
| `07_csf_olink/` | Project 277 CSF Olink HT; PC1 score; 4-way decoupling test (brain vs plasma vs CSF Olink vs Qalb) | `results/step7/decoupling_4way.tsv` |
| `figures/` | Render all main + supplementary figures from upstream outputs | `results/figures/Fig*.png`, `.pdf` |
| `manuscript/83_assemble_supp_tables.py` | Assemble Supplementary Tables S1-S8 into one workbook | `supp_tables/*.xlsx` |

Scripts are intended to be run in numerical order. Each script reads from earlier `results/` outputs and writes its own outputs to a stage-specific subfolder.

Example:

```bash
conda activate pd-bbb
cd 01_foundation && python 01_build_canonical_clinical.py
cd ../02_brain && python 20a_load_smajic.py    # then 20b, 20c, 20d, 21, 22, 30
Rscript 31_deseq2_integrated.R
Rscript 32_wgcna_bootstrap.R
# ... continue through stages 03-07, then figures and manuscript
```

---

## Pre-registration

The full pre-registered analysis plan (including primary endpoint specification, secondary endpoints, subgroups, multiple-comparison strategy, and a complete deviation log) is reproduced verbatim as **Supplementary Table S7** in the manuscript.

Three post-hoc additions are logged in the deviation log:
1. Strengthening analyses (Stage 4): cross-modal decoupling, brain-derived sensitivity model, corrected genotype-call parsing, multi-database functional enrichment, literature effect-size comparison
2. CSF/serum albumin ratio (Qalb) analysis (Stage 6)
3. CSF Olink Explore HT 4-way decoupling (Stage 7)

Each post-hoc design document was committed to version control before the corresponding modelling was run.

---

## Reproducibility notes

- All analyses use a fixed random seed where applicable (specified in `config.yaml`).
- The pinned conda environment (`environment.yml`) locks all dependencies for byte-identical reproducibility.
- R scripts use `set.seed(42)` at the top.
- Bootstrap iterations: WGCNA module stability uses 100 donor-level bootstrap resamples; consensus clustering uses 1,000 bootstrap iterations.

---

## Citation

If you use this code, please cite the paper:

> Atif Z., Akber U., Kim H.-I., Kwon H.-S., Park C.-S. (2026). *Brain, blood, and cerebrospinal fluid vascular biomarkers reflect distinct biology in Parkinson's disease.* npj Parkinson's Disease, [volume]([issue]), [page].

---

## License

This code is released under the MIT License. See `LICENSE` for details.

---

## Contact

Corresponding authors:
- Hyuk-Sang Kwon — hyuksang@gist.ac.kr
- Chul-Seung Park — cspark@gist.ac.kr

For data access questions, contact PPMI directly via https://www.ppmi-info.org.

---

## Acknowledgements

Data used in the preparation of this work were obtained from the Parkinson's Progression Markers Initiative (PPMI) database (RRID:SCR_006431), funded by The Michael J. Fox Foundation for Parkinson's Research and its funding partners. We thank the PPMI study participants and their families, the PPMI investigators and site staff, and the contributors of the publicly available human midbrain single-nucleus RNA-sequencing datasets (Kamath et al., Smajic et al., and Agarwal et al.) used in this study.
