library(FactoMineR)
data(tea)

res.mca = MCA(tea, quanti.sup=19, quali.sup=c(20:36),graph = FALSE)
summary(res.mca)
dimdesc(res.mca)

anova(res.mca$ind$coord[,1]~tea[,1])

library(FactoMineR)
women_work=read.table("http://factominer.free.fr/classical/datasets/women_work.txt", header=TRUE, row.names=1, sep="\t")
res.ca = CA(women_work, col.sup=4:ncol(women_work))

library("FactoMineR")
res.ca <- CA(housetasks, graph = FALSE)
dimdesc(res.ca)
# PCA
data("decathlon2")
decathlon2.active <- decathlon2[1:23, 1:10]
res.pca <- PCA(decathlon2, ind.sup = 24:27, 
               quanti.sup = 11:12, quali.sup = 13, graph=FALSE)
p = dimdesc(res.pca)

library("FactoMineR")
data(wine)
df <- wine[,c(1,2, 16, 22, 29, 28, 30,31)]
head(df[, 1:7], 4)

library(FactoMineR)
res.famd <- FAMD(df, graph = FALSE)
dimdesc(res.famd)
fviz_contrib(res.famd,choice = "ind")


library(FactoMineR)
data(tea)
res.hcpc = HCPC(res.mca)
res.hcpc$desc.var$test.chi2
res.hcpc$desc.var$category
dimdesc(res.hcpc)




