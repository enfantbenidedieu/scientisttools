
################################################################################
# MFA with continues variables
################################################################################

# Multiple Factor Analysis
rm(list = ls())
library(FactoMineR)
library(factoextra)
data(wine)
dim(wine)

res.pca <- PCA(wine,quali.sup = c(1:2),quanti.sup = c(30,31),graph = F)

res.pca$svd$vs
res.pca$svd$U
res.pca$svd$V

res.pca$quanti.sup$cor
