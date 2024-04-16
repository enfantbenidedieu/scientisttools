
library(FactoMineR)
data(tea)
res.mca = MCA(tea, ncp=20, quanti.sup=19, quali.sup=c(20:36), graph=FALSE)

res.hcpc = HCPC(res.mca)