


library(PCAmixdata)

dat <- cbind(gironde$employment,gironde$housing,gironde$services,gironde$environment) 
index <- c(rep(1,9),rep(2,5),rep(3,9),rep(4,4)) 
names <- c("employment","housing","services","environment") 
res.mfamix<-MFAmix(data=dat,groups=index,
                   name.groups=names,ndim=3,rename.level=TRUE,graph=FALSE)

print(res.mfamix)

dat1 <- cbind(gironde$employment,gironde$housing) 
index1 <- c(rep(1,9),rep(2,5)) 
names1 <- c("employment","housing") 
res.mfamix1 <- MFAmix(data=dat1,groups=index1,
                   name.groups=names1,ndim=3,rename.level=TRUE,graph=FALSE)

index.sup1 <- c(rep(3,9),rep(4,4))
names.sup1 <- c("services","environment")
data.sup1 <- cbind(gironde$services,gironde$environment)
supvarpred <- supvar(obj = res.mfamix1,
                     data.sup = data.sup1,
                     groups.sup = index.sup1,
                     name.groups.sup = names.sup1,
                     rename.level=TRUE)


#### PCAMIX
datamix <- cbind(gironde$employment,gironde$housing)
X.quanti <- splitmix(datamix)$X.quanti
X.quali <- splitmix(datamix)$X.quali
res.pcamix <- PCAmix(X.quanti = X.quanti,X.quali = X.quali,ndim = 5,graph = F)

head(res.pcamix$eig)
Xsup.quanti <- splitmix(data.sup1)$X.quanti
Xsup.quali <- splitmix(data.sup1)$X.quali

# pcamix.supvar <- supvar(res.pcamix,X.quanti.sup = Xsup.quanti,X.quali.sup = Xsup.quali,rename.level = TRUE)



