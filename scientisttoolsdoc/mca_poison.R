rm(list = ls())
library(FactoMineR)
library(factoextra)
data (poison)
res.mca <- MCA(poison,ind.sup = c(51:55),quanti.sup = c(1:2),quali.sup = c(3:4),graph = FALSE)

res.mca$call$marge.col
res.mca$call$marge.row
res.mca$svd$vs
res.mca$svd$U
res.mca$svd$V

head(res.mca$ind$coord)

res.mca$quali.sup$eta2

res.mca$quanti.sup$coord

head(res.mca$var$coord)
unique(poison$Sick)
unique(poison$Sex)


data(geomorphology)
res <- FAMD(geomorphology,graph = FALSE)

head(res$quanti.var$coord)
head(res$var$coord)
