
rm(list = ls())
library(FactoMineR)
library(PCAmixdata)
library(dplyr)
data(wine)

X.quanti <- select(wine,where(is.numeric))
X.quali <- select(wine,where(is.character)|where(is.factor))

# recodequanti <- recodquant(X.quanti)
# recodequali <- recodqual(X.quali)


# rec1_quanti = recod(X.quanti = X.quanti,X.quali = NULL)
# rec2_quali = recod(X.quanti = NULL,X.quali = X.quali)
res3 = recod(X.quanti = X.quanti,X.quali = X.quali)

res.pcamix <- PCAmix(X.quanti = X.quanti,X.quali=X.quali,graph = F)
pred <- predict(res.pcamix,X.quanti = X.quanti,X.quali = X.quali)
sup.quanti <- supvar(res.pcamix,X.quanti.sup = X.quanti,X.quali.sup = X.quali)

# res.pcamix <- PCAmix(X.quanti = X.quanti,X.quali=X.quali,graph = F)
# res.famd <- FAMD(wine,graph = F)
res.pca <- PCAmix(X.quanti=X.quanti,X.quali = NULL,graph = F)
res.mca <- MCA(X.quali,graph = F)
res.pca2 = PCAmix(X.quanti = NULL,X.quali = X.quali,graph = F)

pred = predict(res.pca,X.quanti = X.quanti,X.quali = NULL)
pred2 <- predict(res.pca2,X.quanti = NULL,X.quali = X.quali)

res.famd.pca <- FAMD(X.quanti)


data(gironde)
save(gironde,file = "./donnee/gironde.rda")
