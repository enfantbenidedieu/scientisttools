
rm(list = ls())
library(FactoMineR)
library(factoextra)

data(mortality)
res<-MFA(mortality,group=c(9,9),type=c("f","f"),
         name.group=c("1979","2006"),graph = F)

fviz_contrib(res,choice = "partial.axes")

# Eigenvalues
head(res$eig)

###################################### row informations
#######################################################
# Individuals informations
#######################################################
head(res$ind$coord)
head(res$ind$contrib)
head(res$ind$cos2)
head(res$ind$coord.partiel)
head(res$ind$within.inertia)
head(res$ind$within.partial.inertia)

#############################################################
freq <- res$freq
head(freq$coord)
head(freq$contrib)
head(freq$cos2)

######################## Partial axes
partial_axes <- res$partial.axes
partial_axes$coord
partial_axes$contrib
round(partial_axes$cor.between,4)

######################### group informations
group <- res$group
group$coord
group$cos2
group$contrib
group$dist2
group$correlation
group$Lg
group$RV

res$inertia.ratio
