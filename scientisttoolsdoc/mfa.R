
################################################################
# MFA with continues variables
################################################################

# Multiple Factor Analysis
rm(list = ls())
library(FactoMineR)
library(factoextra)
data(wine)
res.mfa <- MFA(wine, 
               ncp  = 5,
               group = c(2, 5, 3, 10, 9, 2), 
               type = c("n", "s", "s", "s", "s", "s"),
               name.group = c("origin","odor","visual",
                              "odor.after.shaking", "taste","overall"),
               num.group.sup = c(1, 6),
               graph = FALSE)

# Separate analyses
res.mfa$separate.analyses$origin
res.mfa$separate.analyses$odor$var$coord
res.mfa$separate.analyses$visual$eig
res.mfa$separate.analyses$odor.after.shaking
res.mfa$separate.analyses$taste

# Eigenvalues
eig <- get_eigenvalue(res.mfa)
eig

# Individuals informations
ind <- get_mfa_ind(res.mfa)
head(ind$coord)
head(ind$cos2)
head(ind$contrib)
head(ind$coord.partiel)
head(ind$within.inertia)
head(ind$within.partial.inertia)

# Continues variables informations
quanti.var <- get_mfa_var(res.mfa, "quanti.var")
head(quanti.var$coord)
head(quanti.var$contrib)
head(quanti.var$cos2)
head(quanti.var$cor)

# Supplementary continue variables
res.mfa$quanti.var.sup$coord
res.mfa$quanti.var.sup$cor
res.mfa$quanti.var.sup$cos2

# Supplementary qualitatives variables
quali_var_sup <- get_mfa_var(res.mfa,"quali.var")
quali_var_sup$coord
quali_var_sup$cos2
quali_var_sup$v.test
head(quali_var_sup$coord.partiel)

head(quali_var_sup$within.inertia) # A implémnter
head(quali_var_sup$within.partial.inertia) # A implémenter

# Group informations
group <- get_mfa_var(res.mfa, "group")
head(group$coord)
head(group$contrib)
head(group$cos2)
head(group$correlation)
round(group$Lg,4)
round(group$RV,4)
group$dist2

# Supplementary group infos
head(group$coord.sup)
head(group$cos2.sup)
group$dist2.sup

# Inertia Informations
res.mfa$inertia.ratio

# Partial axes
partial_axes <- get_mfa_partial_axes(res.mfa)
head(partial_axes$coord)
head(partial_axes$cor)
head(partial_axes$contrib,10)
head(partial_axes$cor.between,10)

#  Summary variables
res.mfa$summary.quanti
res.mfa$summary.quali

# Dimension description
DimDesc <- dimdesc(res.mfa)
DimDesc$Dim.1$quanti
DimDesc$Dim.1$category
DimDesc$Dim.2$quanti
DimDesc$Dim.2$quali
DimDesc$Dim.2$category
DimDesc$Dim.3$quanti

active <- wine[,colnames(res.mfa$call$XTDC)]
predict_mfa <- predict(res.mfa,newdata = wine)

library(PCAmixdata)
data("gironde")
gironde <- 
  cbind.data.frame(
    gironde$employment,
    gironde$housing,
    gironde$services,
    gironde$environment
  )
save(gironde,file = "./donnee/gironde.rda")

employment <- gironde$employment
housing <- gironde$housing
services <- gironde$services
environment <- gironde$environment
save(employment,file = "./donnee/gironde_employment.rda")
save(housing,file = "./donnee/gironde_housing.rda")
save(services,file = "./donnee/gironde_services.rda")
save(environment,file = "./donnee/gironde_environment.rda")

