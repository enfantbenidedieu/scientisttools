rm(list = ls())
library(FactoMineR)
data("children")

data(housetasks)
# res.ca <- CA(housetasks, graph=FALSE)
children$group <- c(rep("A",5),rep("B",5),rep("C",4),rep(NA,4))
res.ca <- CA (children,row.sup = 15:18,
              quanti.sup = 6:8,quali.sup = 9,graph = FALSE)

X.del <- as.data.frame(res.ca$call$X)
Xtot2 <- res.ca$call$Xtot[-c(15:18),]
quali.sup<- c("group")
X.quali.sup <- Xtot2[,quali.sup]
for (j in 1:length(quali.sup)) {
  Xtot2[,quali.sup[j]] <- droplevels(Xtot2[,quali.sup[j]] , reorder=FALSE)
  print(Xtot2)
  X.quali.sup <- rbind(X.quali.sup, matrix(unlist(by(X.del, 
                                                     Xtot2[, quali.sup[j]], colSums)), ncol = ncol(X.del), byrow = T))
}

matrix(unlist(by(X.del,Xtot2[, quali.sup[1]], colSums)),ncol = ncol(X.del), byrow = T)

res.ca$call$marge.col
res.ca$call$row.w

X <- as.data.frame(t(children))
res.ca <- CA (X,row.sup = 6:8,col.sup = c(15:18),graph = F)




svd_trip = svd.triplet(X,row.w = )

res.ca$svd$vs
res.ca$svd$U
res.ca$svd$V

X <- res.ca$call$X
row.w <- rep(1,nrow(X))
X.row.sup <- children[c(15:18),c(1:5)]
somme.row <- rowSums(X.row.sup)
X.row.sup <- X.row.sup/somme.row

marge.col <- res.ca$call$marge.col

dist2.row <- rowSums(t((t(X.row.sup)-marge.col)^2/marge.col))

X.col.sup <- children[c(1:14),c(6:8)]
X.col.sup <- X.col.sup*row.w
colnames(X.col.sup) <- colnames(Xtot)[col.sup]
somme.col <- colSums(X.col.sup)
X.col.sup <- t(t(X.col.sup)/somme.col)


ncp=5
total <- sum(X*row.w)
F <- as.matrix(X)*(row.w/total)
marge.col <- colSums(F)
marge.row <- rowSums(F)
ncp <- min(ncp, (nrow(X) - 1), (ncol(X) - 1))
Tc <- t(t(F/marge.row)/marge.col) - 1

X <- t(t(Tc)*sqrt(marge.col))*sqrt(marge.row)

s_v_d = svd(X)
s_v_d$d
s_v_d$u/sqrt(marge.row)
s_v_d$v/sqrt(marge.col)

U <- res.ca$svd$U
V <- res.ca$svd$V
eig <- res.ca$eig[,1][1:ncol(U)]
t(t(V)*sqrt(eig))
