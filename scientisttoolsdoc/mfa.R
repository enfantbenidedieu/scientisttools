
################################################################################
# MFA with continues variables
################################################################################

# Multiple Factor Analysis
rm(list = ls())
library(FactoMineR)
library(factoextra)
data(wine)
res.mfa <- MFA(wine, 
               # ncp  = NULL,
               group = c(2, 5, 3, 10, 9, 2), 
               type = c("n", "s", "s", "s", "s", "s"),
               name.group = c("origin","odor","visual",
                              "odor.after.shaking", "taste","overall"),
               num.group.sup = c(1, 6),
               graph = FALSE)

res.mfa$global.pca$call$col.w
res.mfa$global.pca$call$row.w
res.mfa$global.pca$call$row.w.init
res.mfa$global.pca$call$X
res.mfa$global.pca$svd$vs
res.mfa$global.pca$svd$U[c(1:5),]
res.mfa$global.pca$svd$V[c(1:5),]

fviz_contrib(res.mfa,choice ="partial.axes")
fviz_cos2(res.mfa,choice = "partial.axes")

###############################################################
# Eigenvalues
#############################################################
eig <- get_eigenvalue(res.mfa)
eig

################################################################################
# Separate analyses
################################################################################
res.mfa$separate.analyses$origin
res.mfa$separate.analyses$odor$var$coord
res.mfa$separate.analyses$visual$eig
res.mfa$separate.analyses$odor.after.shaking
res.mfa$separate.analyses$taste

res.mfa$call$col.w
################################################################################
# Individuals informations
################################################################################

# Individuals
ind <- get_mfa_ind(res.mfa)
head(ind$coord)
head(ind$cos2)
head(ind$contrib)
head(ind$coord.partiel)
head(ind$within.inertia)
head(ind$within.partial.inertia)

################################################################################
# Continues variables informations
################################################################################
# Active Variables
quanti.var <- get_mfa_var(res.mfa, "quanti.var")
head(quanti.var$coord)
head(quanti.var$cos2)
head(quanti.var$contrib)
head(quanti.var$cor)

# Supplementary continue variables
res.mfa$quanti.var.sup$coord
res.mfa$quanti.var.sup$cor
res.mfa$quanti.var.sup$cos2


###############################################################################
# Group informations
###############################################################################

# Group of variables
group <- get_mfa_var(res.mfa, "group")
head(group$coord)
head(group$contrib)
round(group$Lg,4)
group$RV
head(group$correlation)
group$dist2  
group$cos2   

###### Supplementary group infos
head(group$coord.sup)
group$dist2.sup
head(group$cos2.sup)

#############################################################
# Partial axes
##############################################################
partial_axes <- get_mfa_partial_axes(res.mfa)
head(partial_axes$coord)
head(partial_axes$cor)
head(partial_axes$contrib)
head(partial_axes$cor.between)

################################################################
# Inertia Informations
################################################################
res.mfa$inertia.ratio

################################################################
# Supplementary qualitatives variables
################################################################
quali_var_sup <- get_mfa_var(res.mfa,"quali.var")
quali_var_sup$coord
quali_var_sup$cos2
quali_var_sup$v.test
quali_var_sup$coord.partiel
head(quali_var_sup$within.inertia) # A implémnter
head(quali_var_sup$within.partial.inertia) # A implémenter

#################################################################


fviz_mfa_ind(res.mfa)
plot.MFA(res.mfa)

fviz_mfa_var(res.mfa,col.var = "black")

fviz_mfa_group(res.mfa)
fviz_mfa_axes(res.mfa)

quanti_var <- get_mfa_var(res.mfa,"quanti.var")
quali_var <- get_mfa_var(res.mfa,"quali.var")
groups <- get_mfa_var(res.mfa,"group")
