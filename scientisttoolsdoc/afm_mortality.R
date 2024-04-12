
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

#############################################################
freq <- res$freq
head(freq$coord)
head(freq$contrib)
head(freq$cos2)


res$s