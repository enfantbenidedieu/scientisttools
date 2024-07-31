rm(list = ls())
library("FactoMineR")
library("factoextra")
data(poison)
res.mca <- MCA(poison,quanti.sup = c(1,2),quali.sup = c(3,4),graph = F)
# HCPC
res.hcpc <- HCPC(res =res.mca,nb.clust = 3,graph=F)

res.hcpc$desc.var$category$`2`
res.hcpc$desc.var$test.chi2
res.hcpc$desc.var$quanti.var
res.hcpc$desc.var$quanti

res.hcpc$desc.ind$para
