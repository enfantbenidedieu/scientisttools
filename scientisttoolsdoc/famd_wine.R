
rm(list = ls())
library(FactoMineR)
library(factoextra)

data(wine)
res.famd <- FAMD(wine,ncp = NULL,graph = FALSE)

fviz_contrib(res.famd,choice = "quali.var")

fviz_cos2(res.famd)

