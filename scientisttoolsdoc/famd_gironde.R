
rm(list = ls())
library(FactoMineR)
library(PCAmixdata)
library(missMDA)
data(gironde)
dat <- cbind(gironde$employment,gironde$housing,gironde$services,gironde$environment) 

dat$income[is.na(dat$income)] <- mean(dat$income,na.rm = TRUE)

res.famd <- FAMD(dat,ncp=6,graph=FALSE)

# Eigenvalue
head(res.famd$eig)
