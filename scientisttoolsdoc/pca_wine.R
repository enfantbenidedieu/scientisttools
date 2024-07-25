
################################################################################
# PCA with wine dataset
################################################################################

# Multiple Factor Analysis
rm(list = ls())
library(FactoMineR)
library(factoextra)
data(wine)

res.pca <- PCA(wine,quanti.sup = 30:31,quali.sup = 1:2,graph = F)

DimDesc <- dimdesc(res.pca)
DimDesc$Dim.1$category
DimDesc$Dim.2$quali


res.pca$quali.sup$eta2

fviz_cos2(res.pca,choice = "var")

############ Eigenvalues
head(res.pca$eig)

################ Individuals informations
ind <- get_pca_ind(res.pca)
head(ind$coord)
head(ind$cos2)
head(ind$contrib)
head(ind$dist)

############## Variables informations
res.var <- get_pca_var(res.pca)
head(res.var$coord)
head(res.var$cor)
head(res.var$cos2)
head(res.var$contrib)

################ Supplementary elemenet
quanti_sup <- res.pca$quanti.sup
head(quanti_sup$coord)
head(quanti_sup$cor)
head(quanti_sup$cos2)

################## Supplementary qualitatives
quali_sup <- res.pca$quali.sup
head(quali_sup$coord)
head(quali_sup$cos2)
head(quali_sup$v.test)
head(quali_sup$dist)
head(quali_sup$eta2)


###############################################
# HCPC with wine
####################################################"
res.hc <- HCPC(res.pca,graph = F)

res.hc$desc.var$test.chi2
res.hc$desc.var$category
