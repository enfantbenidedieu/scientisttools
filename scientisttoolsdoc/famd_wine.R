
library(FactoMineR)
library(factoextra)

data(wine)
res.famd <- FAMD(wine,ncp = NULL,graph = FALSE)

res.famd$ind$dist


(0.9307194^2)/0.27634264
(1.0508446^2)/0.35227942


res.famd$quali.sup$coord
res.famd$quali.sup$cos2
res.famd$quali.sup$eta2
res.famd$quali.sup$dist
res.famd$quali.sup$v.test

res.famd$ind$coord

