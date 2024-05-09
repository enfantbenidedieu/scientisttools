
rm(list = ls())
library(FactoMineR)
library(dplyr)
data(wine)
res.gpa <- GPA(wine[,-(1:2)], group=c(5,3,10,9,2),
               name.group=c("olf","vis","olfag","gust","ens"),graph = F)

####
res.gpa$RV
res.gpa$RVs
res.gpa$simi
res.gpa$scaling %>% head()
res.gpa$dep
res.gpa$consensus
res.gpa$Xfin
res.gpa$correlations$`cor olf`
res.gpa$PANOVA$objet
