
rm(list = ls())
library(FactoMineR)
library(factoextra)

data(mortality)
res<-MFA(mortality,group=c(9,9),type=c("f","f"),
         name.group=c("1979","2006"),graph = F)

View(res$call$X)
View(res$call$XTDC)

res$ind$coord

###################################### row informations
#######################################################
# Individuals informations
#######################################################
head(res$ind$coord)
head(res$ind$cos2)
head(res$ind$contrib)
head(res$ind$coord.partiel)
res$ind$within.inertia
res$ind$within.partial.inertia

res$eig

res1 <- res$separate.analyses[["1979"]]
View(res1$call$X)
head(res1$call$row.w)
res1$call$col.w
res1$call$scale.unit

View(res$global.pca$call$X)
head(res$global.pca$call$row.w)
res$global.pca$call$col.w
