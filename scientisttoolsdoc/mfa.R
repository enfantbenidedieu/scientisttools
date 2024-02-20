
################################################################################
# MFA with continues variables
################################################################################

# Multiple Factor Analysis
rm(list = ls())
library(FactoMineR)
library(factoextra)
data(wine)
res.mfa <- MFA(wine, 
               group = c(2, 5, 3, 10, 9, 2), 
               type = c("n", "s", "s", "s", "s", "s"),
               name.group = c("origin","odor","visual",
                              "odor.after.shaking", "taste","overall"),
               num.group.sup = c(1, 6),
               graph = FALSE)

################################################################################
# Separate analyses
################################################################################
res.mfa$separate.analyses$origin
res.mfa$separate.analyses$odor
res.mfa$separate.analyses$visual
res.mfa$separate.analyses$odor.after.shaking
res.mfa$separate.analyses$taste
res.mfa$separate.analyses$overall


################################################################################
# Individuals informations
################################################################################

# Individuals
ind <- get_mfa_ind(res.mfa)
head(ind$coord)
head(ind$contrib)
head(ind$cos2)
head(ind$coord.partiel)

### à compléter
head(ind$within.inertia)
head(ind$within.partial.inertia)

  
  data(geomorphology)
################################################################################
# Continues variables informations
################################################################################

# Active Variables
quanti.var <- get_mfa_var(res.mfa, "quanti.var")
# Coordinates
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
group$coord
group$contrib
group$Lg
group$RV
group$correlation

# A chercher
group$dist2  # A implémenter
group$cos2   # A implémenter

###### Supplementary group infos
group$coord.sup
group$cos2.sup # A implémenter
group$dist2.sup # A implémenter

#############################################################
# Partial axes
##############################################################
partial_axes <- get_mfa_partial_axes(res.mfa)
head(partial_axes$coord)
head(partial_axes$cor)
head(partial_axes$contrib) # A implémenter
head(partial_axes$cor.between)

################################################################
# Inertia Informations
################################################################
res.mfa$inertia.ratio

################################################################
# Supplementary qualitatives variables
################################################################
quali_var_sup <- get_mfa_var(res.mfa,"quali.var")
head(quali_var_sup$coord)
head(quali_var_sup$cos2)
head(quali_var_sup$v.test)
head(quali_var_sup$coord.partiel)
head(quali_var_sup$within.inertia) # A implémnter
head(quali_var_sup$within.partial.inertia) # A implémenter

#################################################################
#
res.mfa$summary.quanti
res.mfa$summary.quali


fviz_mfa_var(res.mfa,"quanti.var")
fviz_mfa_var(res.mfa,"group")
fviz_mfa_group(res.mfa)

plot.MFA(res.mfa,choix = "var",graph.type = "classic")
