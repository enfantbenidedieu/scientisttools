#################################################################"
# Muliple Factor Analysis for Contingency Tables (MFACT)
#################################################################

###Example with groups of frequency tables
rm(list = ls())
library(FactoMineR)
data(mortality)
res<-MFA(mortality,group=c(9,9),type=c("f","f"),
         name.group=c("1979","2006"),graph = FALSE)

############################################################
# Separates Analysis
###########################################################

res$separate.analyses

#########################################################
#  Row informations
#########################################################

row_infos <- get_mfa_ind(res)
round(row_infos$coord,3)
row_infos$contrib
row_infos$cos2
row_infos$coord.partiel
row_infos$within.inertia
row_infos$within.inertia
row_infos$within.partial.inertia


res$freq$coord
res$freq$contrib
res$freq$cos2


res$global.pca$eig
res$global.pca$var$coord

row_w = res$call$row.w
col_w = res$call$col.w
X <- res$global.pca$call$X
D = diag(row_w)
M = diag(col_w)

A = t(X)%*%D
A = A%*%as.matrix(X)
A = A%*%as.matrix(M)


s_v_d <- eigen(A)

