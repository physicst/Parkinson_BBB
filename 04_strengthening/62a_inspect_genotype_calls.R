## Step 4 Task 3a  -  Inspect raw LRRK2 / GBA / APOE genotype TESTVALUE strings
## from the PPMI Biospecimen file so we can write a tight, defensible carrier
## call. Print: distinct TESTNAMEs and value frequencies, flag obvious
## mutation-positive vs WT/normal/none patterns.

biospec_path <- "D:/Parkinson_file/Current_Biospecimen_Analysis_Results_02May2026.csv"
b <- read.csv(biospec_path, stringsAsFactors = FALSE)
b$PATNO <- as.character(b$PATNO)
cat(stepf("Loaded biospecimen: %d rows\n", nrow(b)))

cat("\n========== APOE TESTNAMEs ==========\n")
apoe_names <- unique(b$TESTNAME[grepl("APOE|ApoE|apoe", b$TESTNAME)])
print(apoe_names)
apoe_rows <- b[toupper(b$TESTNAME) == "APOE GENOTYPE", ]
cat(stepf("\nAPOE GENOTYPE rows: %d\n", nrow(apoe_rows)))
cat("Distinct TESTVALUEs:\n")
print(table(apoe_rows$TESTVALUE, useNA = "ifany"))

cat("\n\n========== GBA TESTNAMEs ==========\n")
gba_names <- unique(b$TESTNAME[grepl("GBA|gba", b$TESTNAME)])
print(gba_names)
for (tn in gba_names) {
  rs <- b[b$TESTNAME == tn, ]
  cat(stepf("\n--- %s (%d rows) ---\n", tn, nrow(rs)))
  cat("Distinct TESTVALUEs (top 30):\n")
  tab <- sort(table(rs$TESTVALUE, useNA = "ifany"), decreasing = TRUE)
  print(head(tab, 30))
}

cat("\n\n========== LRRK2 TESTNAMEs ==========\n")
lrrk_names <- unique(b$TESTNAME[grepl("LRRK2|lrrk2|Lrrk2", b$TESTNAME)])
print(lrrk_names)
for (tn in lrrk_names) {
  rs <- b[b$TESTNAME == tn, ]
  cat(stepf("\n--- %s (%d rows) ---\n", tn, nrow(rs)))
  cat("Distinct TESTVALUEs (top 30):\n")
  tab <- sort(table(rs$TESTVALUE, useNA = "ifany"), decreasing = TRUE)
  print(head(tab, 30))
}
