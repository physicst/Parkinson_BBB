"""Assemble Supplementary Tables S1-S8 into one .xlsx workbook + companion
pre-registration .md file, formatted for npj Parkinson's Disease submission.

Outputs:
  manuscript/supp_tables/2026-05-25-pd-bbb-supp-tables.xlsx
  manuscript/supp_tables/SuppTableS7_preregistration.md   (copy)
"""
from __future__ import annotations
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

OUT = Path("manuscript/supp_tables")
OUT.mkdir(parents=True, exist_ok=True)
XLSX = OUT / "2026-05-25-pd-bbb-supp-tables.xlsx"


# ----------------------------------------------------------------------
# S1  -  118-gene BBB module with HGNC symbols, hub score, and DESeq2 stats
# ----------------------------------------------------------------------
def build_S1() -> pd.DataFrame:
    mod = pd.read_csv("results/step2/bbb_module_genes.tsv", sep="\t")
    sym = pd.read_csv("results/step4/v2_module_with_symbols.tsv", sep="\t")
    hub = pd.read_csv("results/step4/hub_genes_with_symbols.tsv", sep="\t")
    de = pd.read_csv("results/step2/integrated_de_endothelial.tsv", sep="\t")

    df = mod.merge(sym[["gene_id", "symbol"]], on="gene_id", how="left")
    df = df.merge(hub[["gene_id", "MM", "GS", "hub_score"]],
                  on="gene_id", how="left")
    df = df.merge(
        de[["gene_id", "baseMean", "log2FoldChange", "lfcSE", "stat",
            "pvalue", "padj"]],
        on="gene_id", how="left")
    df = df.rename(columns={"MM": "module_membership_kME",
                            "GS": "gene_significance_GS",
                            "log2FoldChange": "log2FC_PD_vs_Control",
                            "lfcSE": "log2FC_SE",
                            "padj": "padj_BH"})
    df["is_top20_hub"] = df["hub_score"].notna()
    cols = ["gene_id", "symbol", "module", "module_membership_kME",
            "gene_significance_GS", "hub_score", "is_top20_hub",
            "baseMean", "log2FC_PD_vs_Control", "log2FC_SE", "stat",
            "pvalue", "padj_BH"]
    return df[cols].sort_values("hub_score", ascending=False,
                                 na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------
# S2  -  Cross-dataset pseudobulk DESeq2 full results (endo + pericyte)
# ----------------------------------------------------------------------
def build_S2_endothelial() -> pd.DataFrame:
    df = pd.read_csv("results/step2/integrated_de_endothelial.tsv", sep="\t")
    return df.sort_values("padj", na_position="last").reset_index(drop=True)


def build_S2_pericyte() -> pd.DataFrame:
    df = pd.read_csv("results/step2/integrated_de_pericyte.tsv", sep="\t")
    return df.sort_values("padj", na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------
# S3  -  Olink BBB panel composition (20 proteins) + UniProt + rationale
# ----------------------------------------------------------------------
def build_S3() -> pd.DataFrame:
    # categories per Methods §4.5
    panel = [
        ("PECAM1", "Endothelial",
         "Canonical endothelial pan-marker (CD31); junctional integrity."),
        ("CDH5", "Endothelial",
         "VE-cadherin; endothelial adherens-junction marker."),
        ("CD34", "Endothelial",
         "Endothelial / progenitor marker."),
        ("ICAM1", "Endothelial activation / adhesion",
         "Inducible adhesion molecule; canonical activation marker."),
        ("ICAM2", "Endothelial activation / adhesion",
         "Constitutive endothelial adhesion molecule."),
        ("ICAM3", "Endothelial activation / adhesion",
         "Endothelial / leukocyte adhesion molecule."),
        ("VCAM1", "Endothelial activation / adhesion",
         "Vascular cell adhesion molecule; canonical activation marker."),
        ("MCAM", "Endothelial / pericyte",
         "Mural-cell / endothelial marker (CD146)."),
        ("TIE1", "Endothelial",
         "Endothelial receptor tyrosine kinase; angiopoietin pathway."),
        ("SELE", "Adhesion / leukocyte recruitment",
         "E-selectin; endothelial activation, leukocyte rolling."),
        ("SELP", "Adhesion / leukocyte recruitment",
         "P-selectin; platelet / endothelial activation."),
        ("ANGPT1", "Angiocrine / remodelling",
         "Angiopoietin-1; vascular stabilisation."),
        ("ANGPTL4", "Angiocrine / remodelling",
         "Angiopoietin-like 4; vascular permeability regulator."),
        ("MMP9", "Vascular damage / remodelling",
         "Matrix metalloproteinase 9; BBB-disruption marker."),
        ("NOTCH3", "Mural / vascular",
         "Pericyte / smooth-muscle receptor (CADASIL gene)."),
        ("PDGFRA", "Mural / pericyte",
         "Platelet-derived growth factor receptor alpha."),
        ("PDGFRB", "Mural / pericyte",
         "Pericyte canonical receptor (PDGFR-beta)."),
        ("VWF", "Endothelial / coagulation",
         "Von Willebrand factor; endothelial-injury marker."),
        ("NEFL", "Neuronal damage (downstream of BBB)",
         "Neurofilament light; CNS axonal-damage anchor."),
        ("HSPA1A", "Vascular stress (heat-shock cross-modal anchor)",
         "Inducible Hsp70; ties plasma proteomic axis to brain-vascular "
         "heat-shock module."),
    ]
    panel_df = pd.DataFrame(panel,
                            columns=["protein", "category", "rationale"])

    # add UniProt IDs from the Olink Project 9000 NPX file (one of them)
    npx = pd.read_csv(
        "D:/Parkinson_file/PPMI_Project_9000_Plasma_Cardio_NPX_26Apr2026.csv",
        usecols=["UNIPROT", "ASSAY"], low_memory=False).drop_duplicates()
    npx2 = pd.read_csv(
        "D:/Parkinson_file/PPMI_Project_9000_Plasma_NEURO_NPX_26Apr2026.csv",
        usecols=["UNIPROT", "ASSAY"], low_memory=False).drop_duplicates()
    npx3 = pd.read_csv(
        "D:/Parkinson_file/PPMI_Project_9000_Plasma_INF_NPX_26Apr2026.csv",
        usecols=["UNIPROT", "ASSAY"], low_memory=False).drop_duplicates()
    uniprot_map = (pd.concat([npx, npx2, npx3])
                   .drop_duplicates("ASSAY")
                   .set_index("ASSAY")["UNIPROT"].to_dict())
    panel_df["uniprot"] = panel_df["protein"].map(uniprot_map)
    panel_df["olink_platform"] = "Olink Explore 384 (Project 9000, plasma); " \
        "Olink Explore HT (Project 277, CSF)  -  same 20 proteins on both"
    return panel_df[["protein", "uniprot", "category", "rationale",
                     "olink_platform"]]


# ----------------------------------------------------------------------
# S4  -  Per-protein plasma NPX statistics by Vascular endotype
# ----------------------------------------------------------------------
def build_S4() -> pd.DataFrame:
    panel = pd.read_parquet(
        "results/step3/olink_bbb_panel_baseline.parquet")
    score = pd.read_parquet("results/step3/bbb_protein_score.parquet")
    panel["PATNO"] = panel["PATNO"].astype(int)
    score["PATNO"] = score["PATNO"].astype(int)
    merged = panel.merge(score[["PATNO", "vascular_class"]], on="PATNO")
    proteins = [c for c in panel.columns if c != "PATNO"]

    rows = []
    for p in proteins:
        for cls in ["Vascular_High", "Vascular_Low"]:
            v = merged.loc[merged["vascular_class"] == cls, p].dropna()
            rows.append({
                "protein": p, "endotype": cls, "n": len(v),
                "mean_NPX": v.mean(), "sd_NPX": v.std(),
                "median_NPX": v.median(),
                "min_NPX": v.min(), "max_NPX": v.max(),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# S5  -  All pre-registered analyses, full coefficient tables
# ----------------------------------------------------------------------
def _read_lmer_coefs(path: str) -> pd.DataFrame:
    """R lmerTest coefs are saved with terms as rownames (unnamed first col).
    Read with index_col=0 to preserve term names."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = "term"
    return df.reset_index()


def build_S5() -> pd.DataFrame:
    """All model coefficient tables normalised to a common schema:
    {analysis, term, estimate, se, df, t_value, p_value, n_subj, n_obs, notes}.
    """
    rows = []

    # --- 1. Pre-registered primary lmer (61 patients, Olink PC1) ---
    df = _read_lmer_coefs("results/step3/lmer_primary_coefs.tsv")
    for _, r in df.iterrows():
        rows.append({
            "analysis": "Pre-registered primary: UPDRS3 ~ Time * Olink BBB "
                        "PC1 + covariates (lme4, n=61 patients, 187 visits)",
            "term": r["term"],
            "estimate": r["Estimate"], "se": r["Std. Error"],
            "df": r["df"], "t_value": r["t value"],
            "p_value": r["Pr(>|t|)"], "n_subj": 61, "n_obs": 187,
            "notes": "",
        })

    # --- 2. Sensitivity lmer (54 patients, brain-derived score) ---
    df = _read_lmer_coefs("results/step4/lmer_sensitivity_v2_coefs.tsv")
    for _, r in df.iterrows():
        rows.append({
            "analysis": "Sensitivity (post-hoc): UPDRS3 ~ Time * "
                        "brain-derived score + covariates (n=54)",
            "term": r["term"],
            "estimate": r["Estimate"], "se": r["Std. Error"],
            "df": r["df"], "t_value": r["t value"],
            "p_value": r["Pr(>|t|)"], "n_subj": 54, "n_obs": None,
            "notes": "Singular random-intercept variance (noted in figure).",
        })

    # --- 3. Pre-registered secondaries (MoCA, DaT-SCAN) ---
    df = pd.read_csv("results/step3/secondary_analyses_summary.tsv",
                     sep="\t")
    for _, r in df.iterrows():
        rows.append({
            "analysis": "Pre-registered secondaries",
            "term": f"Time * BBB_score_z on {r['endpoint']}",
            "estimate": r["estimate"], "se": r["se"],
            "df": None, "t_value": None,
            "p_value": r["p"], "n_subj": r.get("n_subj"),
            "n_obs": r.get("n_obs"), "notes": "",
        })

    # --- 4. Pre-registered subgroups (original, before genotype fix) ---
    df = pd.read_csv("results/step3/subgroup_analyses_summary.tsv",
                     sep="\t")
    for _, r in df.iterrows():
        rows.append({
            "analysis": "Pre-registered subgroups (original)",
            "term": r.get("label", ""),
            "estimate": r["estimate"], "se": r["se"],
            "df": None, "t_value": None, "p_value": r["p"],
            "n_subj": r.get("n_subj"), "n_obs": r.get("n_obs"),
            "notes": r.get("bh_padj_note", ""),
        })

    # --- 5. Subgroups with corrected genotype calls ---
    df = pd.read_csv("results/step4/subgroup_analyses_fixed.tsv",
                     sep="\t")
    for _, r in df.iterrows():
        rows.append({
            "analysis": "Subgroups with corrected genotype calls (post-hoc)",
            "term": r.get("label", ""),
            "estimate": r["estimate"], "se": r["se"],
            "df": None, "t_value": None, "p_value": r["p"],
            "n_subj": r.get("n_subj"), "n_obs": r.get("n_obs"),
            "notes": "LRRK2 and GBA carriers contained 0 patients in the "
                     "n=61 cohort and were dropped.",
        })

    # --- 6. Qalb longitudinal (post-hoc) ---
    df = pd.read_csv("results/step6/qalb_lmer_coefs.tsv", sep="\t")
    for _, r in df.iterrows():
        rows.append({
            "analysis": "Qalb longitudinal (post-hoc): UPDRS3 ~ Time * "
                        "predictor + covariates",
            "term": f"{r['model']} -- {r['term']}",
            "estimate": r["estimate"], "se": r["se"], "df": r.get("df"),
            "t_value": r.get("t"), "p_value": r["p"],
            "n_subj": r.get("n_subj"), "n_obs": r.get("n_obs"),
            "notes": "Singular fit" if r.get("singular") else "",
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# S6  -  Genotype carrier-call comparison: buggy regex vs corrected
# ----------------------------------------------------------------------
def build_S6() -> pd.DataFrame:
    """Three-way carrier-call comparison per gene:
      1. Original (buggy) permissive regex on the FULL PPMI biospecimen.
      2. Corrected per-SNP rules on the FULL PPMI biospecimen.
      3. The same corrected calls restricted to the n=61 longitudinal
         modelling cohort (which contained 0 LRRK2 and 0 GBA carriers).
    """
    bio = pd.read_csv(
        "D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv",
        low_memory=False, usecols=["PATNO", "TESTNAME", "TESTVALUE"])

    # --- buggy regex (full cohort) ---
    buggy_lrrk2 = set(bio.loc[
        bio["TESTNAME"].astype(str).str.contains("LRRK2", regex=False,
                                                  na=False), "PATNO"])
    buggy_gba = set(bio.loc[
        bio["TESTNAME"].astype(str).str.contains("GBA", regex=False,
                                                  na=False), "PATNO"])

    # --- corrected per-SNP rules (full cohort) ---
    def _non_wt(testname_token: str, wt_values: set) -> set:
        sub = bio[bio["TESTNAME"].astype(str).str.contains(
            testname_token, case=False, na=False)]
        vals = sub["TESTVALUE"].astype(str).str.strip()
        return set(sub.loc[~vals.isin(wt_values | {"", "nan", "NA", "-"}),
                            "PATNO"])

    lrrk2_g2019s = _non_wt("G2019S", {"G/G"})
    lrrk2_r1441h = _non_wt("R1441H", {"G/G"})
    lrrk2_y1699c = _non_wt("Y1699C", {"A/A"})
    lrrk2_i2020t = _non_wt("I2020T", {"T/T"})
    corrected_lrrk2_full = (lrrk2_g2019s | lrrk2_r1441h |
                             lrrk2_y1699c | lrrk2_i2020t)
    n_lrrk2_genotyped = bio.loc[bio["TESTNAME"].astype(str).str.contains(
        "G2019S", case=False, na=False), "PATNO"].nunique()

    # GBA N370S  -  exact-match TESTNAME (the biospecimen file contains a
    # second variant `p.N370S_rs76763715` from a separate assay batch
    # which we exclude to match the original Step 4 R pipeline used
    # for the manuscript narrative). WT in this assay is "T/T".
    n370s = bio[bio["TESTNAME"] == "rs76763715_GBA_p.N370S"]
    n370s_vals = n370s["TESTVALUE"].astype(str).str.strip()
    n370s_carriers = set(n370s.loc[
        ~n370s_vals.isin({"T/T", "", "nan", "NA", "-"}), "PATNO"])

    # GBA full-gene sequencing  -  any value other than WT/WT and
    # 'Non pathogenic copy number loss' counts as a carrier.
    gba_seq = bio[bio["TESTNAME"] == "Full GBA gene sequencing"]
    gba_seq_vals = gba_seq["TESTVALUE"].astype(str).str.strip()
    gba_seq_carriers = set(gba_seq.loc[
        ~gba_seq_vals.isin({"WT/WT", "Non pathogenic copy number loss",
                             "", "nan", "NA", "-"}), "PATNO"])
    corrected_gba_full = n370s_carriers | gba_seq_carriers
    n_gba_genotyped = (bio.loc[bio["TESTNAME"].isin(
        ["rs76763715_GBA_p.N370S", "Full GBA gene sequencing"]),
        "PATNO"].nunique())

    # --- modelling-cohort restriction (n=61) ---
    cohort = pd.read_parquet(
        "results/step3/modeling_cohort_subjects.parquet")
    cohort_patnos = set(cohort["PATNO"].astype(int))
    corrected_lrrk2_modelling = corrected_lrrk2_full & cohort_patnos
    corrected_gba_modelling = corrected_gba_full & cohort_patnos

    rows = [
        {"gene": "LRRK2",
         "method": "Original buggy permissive regex grepl('LRRK2')",
         "cohort_scope": "Full PPMI biospecimen file",
         "n_carrier": len(buggy_lrrk2),
         "note": "Sweeps up any TESTNAME containing the substring "
                 "'LRRK2' -- incorrectly including CSF protein assays, "
                 "fluorospot panels and gene-expression replicates."},
        {"gene": "LRRK2",
         "method": "Corrected per-SNP rules",
         "cohort_scope": f"Full PPMI genotyped (n = {n_lrrk2_genotyped})",
         "n_carrier": len(corrected_lrrk2_full),
         "note": "Non-wildtype genotype at any of rs34637584 (G2019S), "
                 "rs34995376 (R1441H), rs35801418 (Y1699C), or rs35870237 "
                 "(I2020T). Effectively all are G2019S heterozygotes."},
        {"gene": "LRRK2",
         "method": "Corrected per-SNP rules",
         "cohort_scope": "Longitudinal modelling cohort (n=61)",
         "n_carrier": len(corrected_lrrk2_modelling),
         "note": "Subset of the corrected carriers above intersected with "
                 "the n=61 modelling cohort; subgroup test was therefore "
                 "not run."},
        {"gene": "GBA",
         "method": "Original buggy permissive regex grepl('GBA')",
         "cohort_scope": "Full PPMI biospecimen file",
         "n_carrier": len(buggy_gba),
         "note": "Sweeps up any TESTNAME containing the substring 'GBA', "
                 "inflating carrier counts."},
        {"gene": "GBA",
         "method": "Corrected per-SNP rules",
         "cohort_scope": f"Full PPMI genotyped (n = {n_gba_genotyped})",
         "n_carrier": len(corrected_gba_full),
         "note": "Heterozygous/homozygous at rs76763715 (N370S), OR any "
                 "pathogenic call from Full GBA gene sequencing other "
                 "than 'WT/WT' or 'Non pathogenic copy number loss'."},
        {"gene": "GBA",
         "method": "Corrected per-SNP rules",
         "cohort_scope": "Longitudinal modelling cohort (n=61)",
         "n_carrier": len(corrected_gba_modelling),
         "note": "Subset of the corrected carriers above intersected with "
                 "the n=61 modelling cohort; subgroup test was therefore "
                 "not run."},
    ]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# S8  -  CSF Olink HT PC1 per-protein loadings (already exists)
# ----------------------------------------------------------------------
def build_S8() -> pd.DataFrame:
    return pd.read_csv(
        "results/step7/csf_olink_per_protein_loadings.tsv", sep="\t")


# ----------------------------------------------------------------------
# Cover sheet
# ----------------------------------------------------------------------
def cover() -> pd.DataFrame:
    return pd.DataFrame([
        {"Table": "S1", "Title": "Brain-derived BBB module  -  full 118-gene "
         "table",
         "Rows": "118 genes × {gene_id, HGNC symbol, module, kME, GS, hub "
                 "score, is_top20_hub, DESeq2 stats}"},
        {"Table": "S2", "Title": "Cross-dataset pseudobulk DESeq2  -  full "
         "results",
         "Rows": "All tested genes from the integrated pseudobulk DESeq2; "
                 "endothelial and pericyte cell types as separate sheets "
                 "(S2_endothelial, S2_pericyte)."},
        {"Table": "S3", "Title": "Olink BBB panel composition + rationale",
         "Rows": "20 BBB-panel proteins × {UniProt, category, rationale, "
                 "platform}. Same 20-protein panel measured on plasma "
                 "(Olink Explore 384, Project 9000) and CSF (Olink Explore "
                 "HT, Project 277)."},
        {"Table": "S4", "Title": "Per-protein plasma NPX statistics by "
         "Vascular endotype",
         "Rows": "20 proteins × 2 endotypes (Vascular-High, Vascular-Low) "
                 "× {n, mean, SD, median, min, max NPX}."},
        {"Table": "S5", "Title": "All pre-registered (and post-hoc) "
         "analyses  -  full coefficient tables",
         "Rows": "Concatenated coefficient tables for: pre-registered "
                 "primary, post-hoc sensitivity, secondaries, original "
                 "subgroups, corrected-genotype subgroups, and Qalb "
                 "longitudinal models. Each row is one model coefficient."},
        {"Table": "S6", "Title": "Genotype carrier-call comparison (buggy "
         "regex vs corrected)",
         "Rows": "LRRK2 and GBA carrier counts under the original "
                 "permissive grepl() regex versus the corrected per-SNP "
                 "rules described in Methods §4.6."},
        {"Table": "S7", "Title": "Pre-registration full text + deviation "
         "log (verbatim)",
         "Rows": "Provided as the companion file "
                 "SuppTableS7_preregistration.md in this folder."},
        {"Table": "S8", "Title": "CSF Olink Explore HT BBB panel  -  "
         "per-protein PC1 loadings",
         "Rows": "20 proteins × {PC1 loading}; PC1 explains 49 % of CSF "
                 "panel variance; all 20 loadings positive (single shared "
                 "CSF endothelial-activation axis)."},
    ])


# ----------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------
def main() -> None:
    print("Building supplementary tables...")
    sheets: dict[str, pd.DataFrame] = {
        "Cover": cover(),
        "S1_BBB_module": build_S1(),
        "S2_endothelial": build_S2_endothelial(),
        "S2_pericyte": build_S2_pericyte(),
        "S3_Olink_panel": build_S3(),
        "S4_NPX_by_endotype": build_S4(),
        "S5_model_coefficients": build_S5(),
        "S6_genotype_calls": build_S6(),
        "S7_see_companion_md": pd.DataFrame([{
            "Note": "The pre-registration full text + deviation log is "
                    "provided verbatim as the companion file "
                    "SuppTableS7_preregistration.md in this folder."}]),
        "S8_CSF_Olink_loadings": build_S8(),
    }
    for name, df in sheets.items():
        print(f"  {name:24s} {df.shape[0]:>6} rows x {df.shape[1]} cols")

    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"\n-> {XLSX}")

    # Companion S7: copy the pre-registration verbatim
    src = Path("preregistration/2026-05-01-pd-bbb-prereg.md")
    dst = OUT / "SuppTableS7_preregistration.md"
    shutil.copy2(src, dst)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
