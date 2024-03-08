rm(list = ls())
library("FactoMineR")
library(factoextra)
D <- cbind.data.frame(V1=c(218,218,30,5),
                      V2=c(30,23,32,12),
                      V3 = c(10,15,17,14))
rownames(D) <- c("P1.K1","P2.K1","P2.K2","P2.K3")
res.pca <- PCA(D,scale.unit = TRUE, graph = FALSE,row.w = c(1,0.1,0.88,0.02))

res.pca <- PCA(decathlon2,ind.sup = c(24:27),quali.sup = 13,quanti.sup = c(11:12),graph = FALSE)


#########################3
res.pca$eig
