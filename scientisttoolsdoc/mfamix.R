# MFAMIX - gironde dataset
rm(list = ls())
library(PCAmixdata)
data(gironde)

dat <- cbind(gironde$employment,gironde$housing,gironde$services,gironde$environment) 
index <- c(rep(1,9),rep(2,5),rep(3,9),rep(4,4)) 
names <- c("employment","housing","services","environment") 
active <- dat[c(1:500),]
res.mfamix<-MFAmix(data=active,groups=index,
                   name.groups=names,ndim=3,rename.level=TRUE,graph=FALSE)

# Eigenvalues
res.mfamix$eig

# Individuals informations
ind <- res.mfamix$ind
head(ind$coord)
head(ind$cos2)
head(ind$contrib.pct)
head(res.mfamix$ind.partial$employment)
head(res.mfamix$ind.partial$housing)
head(res.mfamix$ind.partial$services)
head(res.mfamix$ind.partial$environment)

 # Quantitatives variables informations
quanti_var <- res.mfamix$quanti
head(quanti_var$coord)
head(quanti_var$cos2)
head(quanti_var$contrib.pct)

# Qualitatives variables informations
quali_var <- res.mfamix$levels
head(quali_var$coord)
head(quali_var$cos2)
head(quali_var$contrib.pct)

# Group informations
head(res.mfamix$groups$Lg)
head(res.mfamix$groups$RV)
head(res.mfamix$groups$contrib.pct)
res.mfamix$inertia.ratio

ind.sup <- dat[c(501:542),]

predict_ind <- predict(res.mfamix,ind.sup)
head(predict_ind)

# Cas 2
dat2 <- cbind(gironde$employment,gironde$housing,gironde$services)
index2 <- c(rep(1,9),rep(2,5),rep(3,9))
names2 <- c("employment","housing","services") 
res.mfamix2 <- MFAmix(data=dat2,groups=index2,name.groups=names2,ndim=3,rename.level=TRUE,graph=FALSE)

head(res.mfamix2$eig)
# Prediction with last group
sup_var2 <- supvar(res.mfamix2,data.sup = gironde$environment,groups.sup = c(rep(1,4)),name.groups.sup = "environment")
head(sup_var2$quanti.sup$coord)
head(sup_var2$quanti.sup$cos2)
sup_var2$group.sup

# Group informations
res.mfamix2$groups$Lg
res.mfamix2$groups$RV

## Cas 3
# Active
dat3 <- cbind(gironde$employment,gironde$housing,gironde$environment) 
index3 <- c(rep(1,9),rep(2,5),rep(3,4)) 
names3 <- c("employment","housing","environment") 
res.mfamix3 <- MFAmix(data=dat3,groups=index3,name.groups=names3,ndim=3,rename.level=TRUE,graph=FALSE)

head(res.mfamix3$eig)

# Sup
sup_var2 <- supvar(res.mfamix2,data.sup = gironde$environment,groups.sup = c(rep(1,4)),name.groups.sup = "environment")
head(sup_var2$quanti.sup$coord)
head(sup_var2$quanti.sup$cos2)


index.sup1 <- c(rep(3,9),rep(4,4))
names.sup1 <- c("services","environment")
data.sup1 <- cbind(gironde$services,gironde$environment)
supvarpred <- supvar(obj = res.mfamix1,
                     data.sup = data.sup1,
                     groups.sup = index.sup1,
                     name.groups.sup = names.sup1,
                     rename.level=TRUE)




