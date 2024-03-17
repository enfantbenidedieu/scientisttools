
###########################################################################
# MFA with categorical variables
###########################################################################
rm(list = ls())
library(FactoMineR)
library(factoextra)
data (poison)
res2.mfa <- MFA(poison, group=c(2,2,5,6), type=c("s","n","n","n"),
                name.group=c("desc","desc2","symptom","eat"),
                num.group.sup=1:2,graph = FALSE)

################################################################
# Eigenvalues
###############################################################

res2.mfa$eig

###############################################################
# Separates Analysis
##############################################################

res2.mfa$separate.analyses$symptom
res2.mfa$separate.analyses$eat
res2.mfa$separate.analyses$desc
res2.mfa$separate.analyses$desc2

##############################################################
# Individuals informations
################################################################
ind <- get_mfa_ind(res2.mfa)
head(ind$coord)
head(ind$cos2)
head(ind$contrib)
head(ind$coord.partiel)
head(ind$within.inertia)
head(ind$within.partial.inertia)

##############################################################
# Qualitatives variables
##############################################################
quali_var <- get_mfa_var(res2.mfa,"quali.var")
head(quali_var$coord)
head(quali_var$cos2)
head(quali_var$contrib)
head(quali_var$v.test)
head(quali_var$coord.partiel)

head(quali_var$within.inertia) # A compléter
head(quali_var$within.partial.inertia) # A compléter

############################################################
# Group
###########################################################
group <- get_mfa_var(res2.mfa,"group")
head(group$coord)
head(group$contrib)
head(group$correlation)
head(group$Lg)
head(group$RV)
head(group$dist2) 
head(group$cos2)

###### Supplementary group infos
head(group$coord.sup)
group$dist2.sup
head(group$cos2.sup)

#############################################################
# Partial axes
##############################################################
partial_axes <- get_mfa_partial_axes(res2.mfa)
head(partial_axes$coord)
head(partial_axes$cor)
head(partial_axes$contrib)
head(partial_axes$cor.between)

################################################################
# Inertia Informations
################################################################
res2.mfa$inertia.ratio

################################################################
# Supplementary qualitatives variables
################################################################
quali_var_sup <- res2.mfa$quali.var.sup
head(quali_var_sup$coord)
head(quali_var_sup$cos2)
head(quali_var_sup$v.test)
head(quali_var_sup$coord.partiel)

head(quali_var_sup$within.inertia) # A implémnter
head(quali_var_sup$within.partial.inertia) # A implémenter

#################################################################
# Supplementary continues columns
################################################################
quanti_var_sup <- res2.mfa$quanti.var.sup
head(quanti_var_sup$coord)
head(quanti_var_sup$cos2)
head(quanti_var_sup$cor)

###############################################################
# Summary
##############################################################
res2.mfa$summary.quali
res2.mfa$summary.quanti


fct.eta2 <- function(vec,x,weights) {  
  VB <- function(xx) {
    return(sum((colSums((tt*xx)*weights)^2)/ni))
  }
  tt <- tab.disjonctif(vec)
  ni <- colSums(tt*weights)
  unlist(lapply(as.data.frame(x),VB))/colSums(x*x*weights)
}

X.quali.sup <- wine[,c("Label","Soil")]

row.w <- res.mfa$global.pca$call$row.w
t(sapply(X.quali.sup,fct.eta2,res.mfa$ind$coord,weights=row.w))

tt <- tab.disjonctif(X.quali.sup[,c("Label")])
ni <- colSums(tt*row.w)
sum((colSums((tt*res.mfa$ind$coord[,1])*row.w)^2)/ni)

(tt*res.mfa$ind$coord[,1]) *row.w

VB <- function(xx) {
  
  return(sum((colSums((tt*xx)*row.w)^2)/ni))
}

lapply(as.data.frame(res.mfa$ind$coord),VB)
unlist(lapply(as.data.frame(res.mfa$ind$coord),VB))

sum((colSums((tt*res.mfa$ind$coord[,1])*row.w)^2)/ni)

colSums(res.mfa$ind$coord[,1]*res.mfa$ind$coord[,1]*row.w)

sum(res.mfa$ind$coord[,1]*res.mfa$ind$coord[,1]*row.w)

VB(as.data.frame(res.mfa$ind$coord[,1]))


res2.mfa$global.pca$var$coord
res2.mfa$global.pca$quali.sup$coord

res2.mfa$global.pca$quali.sup$cos2
res2.mfa$global.pca$var$cos2
