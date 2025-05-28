library(limma)

ecoli_glycerol <- read.csv("/path/to/ecoli_glycerol_raw.csv", row.names = 1)

group <- factor(c(
  rep("strain_glycerol1", 3),
  rep("strain_glycerol2", 3),
  rep("strain_glycerolA", 3),
  rep("strain_glycerolB", 3),
  rep("strain_glycerolC", 3),
  rep("strain_glycerolD", 3),
  rep("strain_glycerolE", 3),
  rep("wildtype", 5)
))

col_names <- colnames(ecoli_glycerol)

strain1_cols <- col_names[1:3]
strain2_cols <- col_names[4:6]
strainA_cols <- col_names[7:9]
strainB_cols <- col_names[10:12]
strainC_cols <- col_names[13:15]
strainD_cols <- col_names[16:18]
strainE_cols <- col_names[19:21]
wildtype_cols <- col_names[22:26]

strains_cols <- list(
  strain_glycerol1 = strain1_cols,
  strain_glycerol2 = strain2_cols,
  strain_glycerolA = strainA_cols,
  strain_glycerolB = strainB_cols,
  strain_glycerolC = strainC_cols,
  strain_glycerolD = strainD_cols,
  strain_glycerolE = strainE_cols
)

pairwise_results <- list()

for (strain_name in names(strains_cols)) {
  current_cols <- c(strains_cols[[strain_name]], wildtype_cols)
  ecoli_glycerol_subset <- ecoli_glycerol[, current_cols]
  
  sample_groups <- rep(c(strain_name, "wildtype"), times = c(3, 5))
  
  design <- model.matrix(~ 0 + factor(sample_groups))
  colnames(design) <- c(strain_name, "wildtype")
  
  fit <- lmFit(ecoli_glycerol_subset, design)
  
  contrast_matrix <- makeContrasts(contrasts = paste0(strain_name, "-wildtype"), levels = design)
  
  fit2 <- contrasts.fit(fit, contrast_matrix)
  fit2 <- eBayes(fit2)
  
  pairwise_results[[paste0(strain_name, "_vs_wildtype")]] <- topTable(fit2, coef = 1, number = Inf, adjust.method = "BH")
}

logFC_values <- list()
pval_values <- list()

for (pair in names(pairwise_results)) {
  logFC_values[[pair]] <- pairwise_results[[pair]]$logFC
  pval_values[[pair]] <- pairwise_results[[pair]]$adj.P.Val
}

logFC_values <- do.call(cbind, logFC_values)
pval_values <- do.call(cbind, pval_values)
rownames(logFC_values) <- rownames(pairwise_results[[1]])
rownames(pval_values) <- rownames(pairwise_results[[1]])
colnames(logFC_values) <- paste0(names(pairwise_results), "_logFC")
colnames(pval_values) <- paste0(names(pairwise_results), "_pval")

combined_results <- data.frame(logFC_values, pval_values)
