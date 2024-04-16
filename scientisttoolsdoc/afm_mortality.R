
rm(list = ls())
library(FactoMineR)
library(factoextra)

data(mortality)
res<-MFA(mortality,group=c(9,9),type=c("f","f"),
         name.group=c("1979","2006"),graph = F)

fviz_contrib(res,choice = "partial.axes")

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


######################## Partial axes
partial_axes <- res$partial.axes
partial_axes$coord
partial_axes$contrib
round(partial_axes$cor.between,4)

######################### group informations
group <- res$group
group$coord
group$dist2

summary(res)

fviz_mfa_ind(res)
fviz_mfa_var(res,choice = "freq")

fviz_mfa_axes(res)
plot.MFA(res,choix = "freq",invisible = "ind")

################################################################
# Separate analysis

sep1 = res$separate.analyses$`1979`
head(sep1$ind$coord)

get_mfa_var(res,element = "freq")



res.mca$var$v.test


library(FactoMineR)
# Compute PCA with ncp = 3
res.pca <- PCA(USArrests, ncp = 3, graph = FALSE)
# Compute hierarchical clustering on principal components
res.hcpc <- HCPC(res.pca, graph = FALSE)

res.hcpc$desc.var$quanti

