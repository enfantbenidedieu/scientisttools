
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

PCAmix<- function (X.quanti=NULL,X.quali=NULL,ndim=5,rename.level=FALSE,
                   weight.col.quanti=NULL,weight.col.quali=NULL,graph=TRUE)
{
  cl <- match.call()
  
  rec <- recod(X.quanti, X.quali,rename.level)
  n <- rec$n
  p <- rec$p
  p1 <- rec$p1
  p2 <- p - p1
  X <- rec$X
  W <- rec$W
  m <- ncol(W) - p1
  
  indexj <- rec$indexj
  
  #construction of the metrics
  N <- rep(1/n, n)
  M1 <- M2 <- NULL #standard metric for PCAmix
  P1 <- P2 <- NULL #supplementary metric for columns like the weights from MFAmix
  if (p1!=0) 
  {
    M1 <- rep(1,p1) 
    P1 <- rep(1,p1) 
    if (!(is.null(weight.col.quanti)))   
    {
      if (length(weight.col.quanti) != ncol(X.quanti))
        stop("the length of \"weight.col.quant\" is different from the number of columns in X.quanti",call. = FALSE)
      P1 <- weight.col.quanti
    }
  }
  
  if (p2!=0)
  {
    ns <- apply(rec$G, 2, sum)
    M2 <- n/ns
    P2 <- rep(1,m)
    if (!(is.null(weight.col.quali)))   
    {
      if (length(weight.col.quali) != ncol(X.quali))
        stop("the length of \"weight.col.quali\" is different from the number of columns in X.quanti",call. = FALSE)
      P2 <- rep(weight.col.quali,rec$nbmoda) 
    }
  }
  M <- c(M1,M2)
  P <- c(P1,P2)
  M.col <- P*M
  names(M.col) <- colnames(W)
  
  #GSVD
  e <- svd.triplet(W, N, M.col)
  V.total.dim <- e$V
  U.total.dim <- e$U
  d.total.dim <- e$vs
  
  #explained inertia
  q <- qr(W)$rank
  eig <- matrix(0, q, 3)
  colnames(eig) <- c("Eigenvalue", "Proportion", "Cumulative")
  rownames(eig) <- paste("dim", 1:q, sep = " ")
  eig[, 1] <- e$vs[1:q]^2
  eig[, 2] <- 100 * eig[, 1]/sum(eig[, 1], na.rm = T)
  eig[1, 3] <- eig[1, 2]
  if (q > 1) 
  {
    for (j in 2:q) eig[j, 3] <- eig[j, 2] + eig[j - 1, 3]
  }
  
  #number of retained dimensions
  if (ndim <= 1) 
    stop("\"ndim\" must be an interger greater or equal to 2",call. = FALSE)
  ndim <- min(ndim, q)
  d <- e$vs[1:ndim]
  
  
  #scores
  U <- e$U[, 1:ndim,drop=FALSE]
  rownames(U) <- rownames(W)
  colnames(U) <- paste("dim", 1:ndim, sep = " ")
  
  F <- sweep(U,2,STATS=d,FUN="*")
  F.total.dim <- sweep(U.total.dim,2,STATS=d.total.dim,FUN="*")
  
  contrib.ind<-F^2/n 
  contrib.ind.pct <- sweep(contrib.ind, 2, STATS = d^2, FUN = "/")
  cos2.ind <- sweep(F^2, 1, STATS = apply(F.total.dim, 1, function(v) {return(sum(v^2))
  }), FUN = "/")
  
  result.ind <- list(coord = F, contrib = contrib.ind, 
                     contrib.pct = 100 * contrib.ind.pct,
                     cos2 = cos2.ind)
  
  #loadings and contributions
  A1 <- A2 <- NULL
  C1 <- C2 <- NULL
  contrib.quanti <- contrib.quali <- NULL
  
  V <- e$V[, 1:ndim,drop=FALSE]
  rownames(V) <- colnames(W)
  colnames(V) <- paste("dim", 1:ndim, sep = " ")
  
  if(p1 >0)
  {
    V1 <- V[1:p1, ,drop=FALSE]
    V1.total.dim <- V.total.dim[1:p1, ,drop=FALSE]
    A1 <- sweep(V1,2,STATS=d,FUN="*")
    A1.total.dim <- sweep(V1.total.dim,2,STATS=d.total.dim,FUN="*")
    #contrib.quanti <- sweep(A1^2, 1, STATS = M1, FUN = "*")
    contrib.quanti <- sweep(A1^2, 1, STATS = M1*P1, FUN = "*")
    contrib.quanti.pct <- sweep(contrib.quanti, 2, STATS = d^2, 
                                FUN = "/")
    cos2.quanti <- sweep(A1^2, 1, STATS = apply(A1.total.dim, 
                                                1, function(v) {
                                                  return(sum(v^2))
                                                }), FUN = "/")
  }
  if(p2 >0)
  {
    V2 <- V[(p1 + 1):(p1 + m), ,drop=FALSE]
    V2.total.dim <- V.total.dim[(p1 + 1):(p1+m), ,drop=FALSE]
    A2 <- sweep(V2,2,STATS=d,FUN="*")
    A2 <- sweep(A2,1,STATS=M2,FUN="*")
    A2.total.dim <- sweep(V2.total.dim,2,STATS=d.total.dim,FUN="*")
    A2.total.dim <- sweep(A2.total.dim,1,STATS=M2,FUN="*")
    contrib.moda <- sweep(A2^2, 1, STATS = ns/n, FUN = "*")
    if (!is.null(weight.col.quali)) 
      contrib.moda <- sweep(contrib.moda, 1, STATS = P2, FUN = "*")
    contrib.moda.pct <- sweep(contrib.moda, 2, STATS = d^2, FUN = "/")
    print(apply(A2.total.dim, 1, function(v) {return(sum(v^2))}))
    cos2.moda <- sweep(A2^2, 1, STATS = apply(A2.total.dim, 1, function(v) {return(sum(v^2))}), FUN = "/")
    contrib.quali <-matrix(NA,p2,ndim)
    rownames(contrib.quali) <- colnames(X.quali)
    colnames(contrib.quali) <- paste("dim", 1:ndim, sep = " ")
    for (j in 1:p2)
    {
      contrib.quali[j,] <- apply(contrib.moda[which(indexj == (j+p1))-p1, ,drop=FALSE], 2, sum)
    }
    contrib.quali.pct<-sweep(contrib.quali, 2, STATS = d^2, FUN = "/")
  }
  
  sqload <- rbind(contrib.quanti, contrib.quali) #correlation ratio
  weight.col <- c(weight.col.quanti,weight.col.quali)
  if (!is.null(weight.col))
    sqload <- sweep(sqload,1,weight.col,"/")
  quali.eta2 <- NULL
  if (p2 > 0) quali.eta2 <- sqload[(p1+1):(p1+p2),,drop=FALSE]
  
  #organization of the results
  result.quanti <- result.levels <- result.quali <- NULL
  if (p1!=0) 
    result.quanti <- list(coord = A1, contrib= contrib.quanti, contrib.pct = 100 * contrib.quanti.pct, 
                          cos2 = cos2.quanti)
  if (p2!=0) 
  {
    result.levels <- list(coord = A2, contrib=contrib.moda, contrib.pct = 100 * contrib.moda.pct, 
                          cos2 = cos2.moda)
    result.quali<-list(contrib = contrib.quali, contrib.pct=contrib.quali.pct*100)
  }
  
  
  
  #coefficient of linear combinaisons defining PC
  
  coef <- list()
  for (i in 1:ndim)
  {
    beta <- V[,i]*M.col
    if (p1 > 0) beta[1:p1] <-  beta[1:p1]/rec$s[1:p1]
    beta0 <- -sum(beta*rec$g)
    coef[[i]] <- as.matrix(c(beta0,beta))
  }
  names(coef) <- paste("dim",1:ndim, sep = "")
  
  #matrix A for PCArot
  A2rot <- NULL
  if (p2 >0) A2rot <- sweep(A2,1,STATS=sqrt(ns/n) ,FUN="*")
  A <- rbind(A1,A2rot) 
  Z <- rec$Z
  res <- list(call = cl, 
              eig = eig, 
              ind = result.ind,
              quanti = result.quanti, 
              levels = result.levels,
              quali=result.quali,
              sqload = sqload, 
              coef = coef, Z = Z, 
              M = M.col,
              quanti.sup=NULL,
              levels.sup=NULL,
              sqload.sup=NULL,
              rec.sup=NULL,
              scores.stand = U, 
              scores = F, 
              V = V, 
              A = A, 
              categ.coord = A2, 
              quanti.cor = A1, 
              quali.eta2 = quali.eta2, 
              rec = rec, 
              ndim = ndim, 
              W = W, 
              rename.level=rename.level)
  class(res) <- "PCAmix"
  if (graph==TRUE) {
    plot.PCAmix(res)
    if (p1 != p) 
      plot.PCAmix(res, choice = "levels")
    if (p1 != 0) 
      plot.PCAmix(res, choice = "cor")
    plot.PCAmix(res, choice = "sqload")
  }
  return(res)
}
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
