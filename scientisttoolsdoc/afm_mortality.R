###############################################################################
# Multiple Factor Analysis for Contingency tables (MFACT)
###############################################################################
rm(list = ls())
library(FactoMineR)
library(factoextra)

# Create dataset with supplementary groups and supplementary individuals
data(mortality)
mortality2 <- mortality
colnames(mortality2) <- c(paste0(colnames(mortality2),"-2"))
dat <- cbind.data.frame(mortality,mortality2)

res <- MFA(dat,group=c(rep(9,4)),type=c(rep("f",4)),
           name.group=c("1979","2006","1979-2","2006-2"),
           ind.sup = c(51:62),num.group.sup = c(3,4),graph = F)

# Eigenvalues
round(res$eig,4)

#######################################################
# Individuals informations
#######################################################
head(res$ind$coord)
head(res$ind$contrib)
head(res$ind$cos2)
head(res$ind$coord.partiel)
head(res$ind$within.inertia)
head(res$ind$within.partial.inertia)

# Supplementary individuals informations
head(res$ind.sup$coord)
head(res$ind.sup$cos2)
head(res$ind.sup$coord.partiel)

#############################################################
freq <- res$freq
head(freq$coord)
head(freq$contrib)
head(freq$cos2)

# Supplementary columns
freq_sup <- res$freq.sup
head(freq_sup$coord)
head(freq_sup$cos2)

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

# Supplementary group
head(res$group$coord.sup)
head(res$group$cos2.sup)
head(res$group$dist2.sup)


dimDesc = dimdesc(res)
head(dimDesc$Dim.1)


plot.MFA(res,choix = "freq")

# Supplementary frequences
mortality2 <- mortality
colnames(mortality2) <- c(paste0(colnames(mortality2),"-2"))

dat <- cbind.data.frame(mortality,mortality2)
res2<-MFA(dat,group=c(rep(9,4)),type=c(rep("f",4)),
         name.group=c("1979","2006","1979-2","2006-2"),
         num.group.sup = c(3,4),graph = F)

dim(mo)
# Group_sup
res2$group$Lg

res2$freq.sup$coord
