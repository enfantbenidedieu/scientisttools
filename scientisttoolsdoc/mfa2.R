
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

fviz_contrib(res2.mfa,choice = "partial.axes")

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

fviz_mfa_ind(res2.mfa)
fviz_mfa_var(res2.mfa,choice = "quali.var")
fviz_mfa_group(res2.mfa)
grp <- as.factor(poison[, "Vomiting"])
fviz_mfa_quali_biplot(res2.mfa, repel = FALSE, col.var = "#E7B800",
                      habillage = grp, addEllipses = TRUE, ellipse.level = 0.95)

fviz_mfa_axes(res2.mfa)

quanti_var <- get_mfa_var(res2.mfa,"quanti.var")

