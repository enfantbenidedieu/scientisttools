

rm(list = ls())
library(FactoMineR)
library(factoextra)
library(PCAmixdata)
data("gironde")
gironde <- 
  cbind.data.frame(
    gironde$employment,
    gironde$housing,
    gironde$services,
    gironde$environment
  )

# gironde$income[is.na(gironde$income)] <- mean(gironde$income, na.rm = TRUE)
res.mfa <- MFA(gironde, 
               ncp  = 5,
               group = c(9,5,9,4), 
               type = c("s","m","n","s"),
               name.group = c("employment","housing","services","environment"),
               num.group.sup = c(2,3),
               ind.sup = c(501:542),
               graph = FALSE)

active <- cbind.data.frame(gironde$employment,gironde$environment,gironde$housing)
res.mfa <- MFA(active, 
               ncp  = 5,
               group = c(9,4,5), 
               type = c("s","s","m"),
               name.group = c("employment","environment","housing"),
               ind.sup = c(501:542),
               num.group.sup = 3,
               graph = FALSE)

eig <- res.mfa$eig
round(eig,4)

head(res.mfa$ind.sup$coord)
head(res.mfa$ind.sup$cos2)
head(res.mfa$ind.sup$coord.partiel)
