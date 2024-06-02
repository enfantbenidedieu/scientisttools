
rm(list = ls())
library(FactoMineR)
library(PCAmixdata)
library(missMDA)
data(gironde)
dat <- cbind(gironde$employment,gironde$housing,gironde$services,gironde$environment) 

# Fill NA with mean
dat$income[is.na(dat$income)] <- mean(dat$income,na.rm = TRUE)

res.famd <- FAMD(dat,ncp=5,graph=FALSE)

# Eigenvalue
head(res.famd$eig)

# Individuals informations
head(res.famd$ind$coord)
head(res.famd$ind$cos2)
head(res.famd$ind$contrib)
head(res.famd$ind$dist)

# Continuous variables informations
head(res.famd$quanti.var$coord)
head(res.famd$quanti.var$cos2)
head(res.famd$quanti.var$contrib)

# Qualitatives/categoricals variables informations
head(res.famd$quali.var$coord)
head(res.famd$quali.var$cos2)
head(res.famd$quali.var$contrib)
head(res.famd$quali.var$v.test)

# Variables informations
head(res.famd$var$coord)
head(res.famd$var$cos2)
head(res.famd$var$contrib)



###### FAMD with supplementary elements

res.famd2<- FAMD(dat,sup.var = c(15:27),ind.sup = c(501:542),graph = FALSE)

# Eigenvalue
head(res.famd2$eig)

# Individuals informations
head(res.famd2$ind$coord)
head(res.famd2$ind$cos2)
head(res.famd2$ind$contrib)
head(res.famd2$ind$dist)

# Continuous variables informations
head(res.famd2$quanti.var$coord)
head(res.famd2$quanti.var$cos2)
head(res.famd2$quanti.var$contrib)

# Qualitatives/categoricals variables informations
head(res.famd2$quali.var$coord)
head(res.famd2$quali.var$cos2)
head(res.famd2$quali.var$contrib)
head(res.famd2$quali.var$v.test)

# Variables informations
res.famd2$var$coord
res.famd2$var$cos2
res.famd2$var$contrib

# Supplementary individuals
head(res.famd2$ind.sup$coord)
head(res.famd2$ind.sup$cos2)
head(res.famd2$ind.sup$dist)

# Supplementary continuous variables
head(res.famd2$quanti.sup$coord)
head(res.famd2$quanti.sup$cor)
head(res.famd2$quanti.sup$cos2)

# Supplementary qualitatives variables
head(res.famd2$quali.sup$coord)
head(res.famd2$quali.sup$cos2)
head(res.famd2$quali.sup$v.test)
head(res.famd2$quali.sup$dist)
head(res.famd2$quali.sup$eta2)
