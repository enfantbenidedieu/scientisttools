rm(list = ls())
library(PCAmixdata)
Biometrie2=read.table("donnee/Biometrie2.csv",header=TRUE,sep=";",dec=".",row.names=1)

index <- c(rep(1,3),rep(2,3)) 
names <- c("GR1","GR2")
res.mfamix<-MFAmix(data=Biometrie2[,c(1:6)],groups=index,
                   name.groups=names,ndim=6,rename.level=TRUE,graph=FALSE)
# Eigenvalues
res.mfamix$eig

# Individuals informations
ind = res.mfamix$ind
ind$coord
ind$cos2
ind$contrib.pct

res.mfamix$ind.partial$GR1
res.mfamix$ind.partial$GR2
