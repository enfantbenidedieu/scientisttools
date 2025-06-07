imputeMCA <- function(don,ncp=2,method=c("Regularized","EM"),row.w=NULL,coeff.ridge=1,threshold=1e-6,
                      ind.sup = NULL, quanti.sup=NULL, quali.sup=NULL, seed=NULL,maxiter=1000){   
  
  moy.p <- function(V, poids) {
    res <- sum(V * poids,na.rm=TRUE)/sum(poids[!is.na(V)])
  }
  find.category <- function (X,tabdisj){
    nbdummy <- rep(1,ncol(X))
    is.quali <- which(!unlist(lapply(X,is.numeric)))
    nbdummy[is.quali] <- unlist(lapply(X[,is.quali,drop=FALSE],nlevels))
    vec = c(0,cumsum(nbdummy))
    Xres <- X
    for (i in is.quali) {
      temp <- as.factor(levels(X[, i])[apply(tabdisj[,(vec[i] + 1):vec[i + 1]], 1, which.max)])
      Xres[,i]<-factor(temp,levels(X[,is.quali][,i]))
    }
    return(Xres)
  }
  
  # tab.disjonctif.NA <- function(tab) {
  # tab <- as.data.frame(tab)
  # modalite.disjonctif <- function(i) {
  # moda <- tab[, i]
  # nom <- names(tab)[i]
  # n <- length(moda)
  # moda <- as.factor(moda)
  # x <- matrix(0, n, length(levels(moda)))
  # ind <- (1:n) + n * (unclass(moda) - 1)
  # indNA <- which(is.na(ind))
  # x[(1:n) + n * (unclass(moda) - 1)] <- 1
  # x[indNA, ] <- NA
  # if ((ncol(tab) != 1) & (levels(moda)[1] %in% c(1:nlevels(moda), "n", "N", "y", "Y"))) dimnames(x) <- list(row.names(tab), paste(nom,levels(moda), sep = "."))
  # else dimnames(x) <- list(row.names(tab), levels(moda))
  # return(x)
  # }
  # if (ncol(tab) == 1) res <- modalite.disjonctif(1)
  # else {
  # res <- lapply(1:ncol(tab), modalite.disjonctif)
  # res <- as.matrix(data.frame(res, check.names = FALSE))
  # }
  # return(res)
  # }
  
  
  ########## Debut programme principal
  don <- as.data.frame(don)
  
  is.quali <- which(!unlist(lapply(don,is.numeric)))
  don[,is.quali] <- lapply(don[,is.quali,drop=FALSE],as.factor)
  
  if (!is.null(which(lapply(don,class)=="logical"))){
    for (k in which(lapply(don,class)=="logical")) don[,k] <- as.factor(don[,k])
  }
  
  method <- match.arg(method,c("Regularized","regularized","EM","em"),several.ok=T)[1]
  method <- tolower(method)
  don <- droplevels(don)
  if (is.null(row.w)) row.w<-rep(1/nrow(don),nrow(don))
  if (!is.null(ind.sup)) row.w[ind.sup] <- 1e-08/length(row.w)   # divide by the number of individuals to be sure that the weight will be small
  Vec <- rep("var",ncol(don))
  NLevels <- sapply(don,nlevels)
  if (!is.null(quali.sup)) Vec[quali.sup] <- "quali.sup"
  if (!is.null(quanti.sup)){
    Vec[quanti.sup] <- "quanti.sup"
    NLevels[NLevels==0] <- 1
  }
  TabDisjMod <- rep(Vec,NLevels)
  
  if (ncp==0) {
    if (is.null(quanti.sup)){
      tab.disj <- tab.disjonctif.prop(don,NULL,row.w=row.w)
      print(head(tab.disj))
      compObs <- find.category(don,tab.disj)
    } else {
      tab.disj <- tab.disjonctif.prop(don[,-quanti.sup],NULL,row.w=row.w)
      compObs <- find.category(don[,-quanti.sup],tab.disj)
      aux <- don
      aux[,-quanti.sup] <- compObs
      compObs <- aux
      
      Tabaux <- cbind.data.frame(tab.disj,don[,quanti.sup,drop=FALSE])
      ordre <- c(which(TabDisjMod!="quanti.sup"),which(TabDisjMod=="quanti.sup"))
      tab.disj <- Tabaux[,order(ordre)]
    }
    return(list(tab.disj=tab.disj,completeObs = compObs))
  }
  
  if (is.null(quanti.sup)) {
    tab.disj.NA <- tab.disjonctif(don)
    tab.disj.comp <- tab.disjonctif.prop(don,seed,row.w=row.w)
  } else {
    tab.disj.NA <- tab.disjonctif(don[,-quanti.sup])
    tab.disj.comp <- tab.disjonctif.prop(don[,-quanti.sup],seed,row.w=row.w)
  }
  hidden <- which(is.na(tab.disj.NA))
  tab.disj.rec.old <- tab.disj.comp
  
  continue <- TRUE
  nbiter <- 0
  
  while (continue){
    
    nbiter <- nbiter+1
    if (length(quali.sup) >0) tab.disj.comp[,TabDisjMod[TabDisjMod!="quanti.sup"]=="quali.sup"] <- tab.disj.comp[,TabDisjMod[TabDisjMod!="quanti.sup"]=="quali.sup"] * 1e-8
    M <- apply(tab.disj.comp, 2, moy.p,row.w)/ncol(don)
    if (any(M<0)) stop(paste("The algorithm fails to converge. Choose a number of components (ncp) less or equal than ",ncp-1," or a number of iterations (maxiter) less or equal than ",maxiter-1,sep=""))
    
    Z <- t(t(tab.disj.comp)/apply(tab.disj.comp, 2, moy.p,row.w))
    Z <- t(t(Z)-apply(Z,2,moy.p,row.w))
    Zscale <- t(t(Z)*sqrt(M))
    
    svd.Zscale <- FactoMineR::svd.triplet(Zscale,row.w=row.w,ncp=ncp)
    moyeig <- 0
    if (length(quanti.sup)+length(quali.sup)>0) NcolZscale <- sum(TabDisjMod=="var")
    else NcolZscale <- ncol(Zscale)
    #  if (nrow(don)>(NcolZscale-ncol(don))) moyeig <- mean(svd.Zscale$vs[-c(1:ncp,(NcolZscale-ncol(don)-length(quali.sup)- length(quanti.sup)+1):NcolZscale)]^2)
    if (nrow(don)>(NcolZscale-ncol(don))) moyeig <- mean(svd.Zscale$vs[-c(1:ncp,(NcolZscale-(ncol(don)-length(quali.sup)-length(quanti.sup))+1):NcolZscale)]^2)
    else moyeig <- mean(svd.Zscale$vs[-c(1:ncp,NcolZscale:length(svd.Zscale$vs))]^2)
    moyeig <- min(moyeig*coeff.ridge,svd.Zscale$vs[ncp+1]^2)
    if (method=="em") moyeig <-0
    eig.shrunk <- ((svd.Zscale$vs[1:ncp]^2-moyeig)/svd.Zscale$vs[1:ncp])
    
    if (ncp==1) rec <- tcrossprod(svd.Zscale$U[,1]*eig.shrunk,svd.Zscale$V[,1])
    else rec <- tcrossprod(t(t(svd.Zscale$U[,1:ncp,drop=FALSE])*eig.shrunk),svd.Zscale$V[,1:ncp,drop=FALSE])
    
    tab.disj.rec <- t(t(rec)/sqrt(M)) + matrix(1,nrow(rec),ncol(rec)) 
    tab.disj.rec <- t(t(tab.disj.rec)*apply(tab.disj.comp,2,moy.p,row.w))
    
    diff <- tab.disj.rec - tab.disj.rec.old
    diff[hidden] <- 0
    relch <- sum(diff^2*row.w)
    tab.disj.rec.old <- tab.disj.rec
    tab.disj.comp[hidden] <- tab.disj.rec[hidden]
    if (length(quali.sup) >0) tab.disj.comp[,TabDisjMod[TabDisjMod!="quanti.sup"]=="quali.sup"] <- tab.disj.comp[,TabDisjMod[TabDisjMod!="quanti.sup"]=="quali.sup"] * 1e+08
    continue=(relch > threshold)&(nbiter<maxiter)
  }
  if (is.null(quanti.sup)){
    compObs <- find.category(don,tab.disj.comp)
  } else {
    compObs <- find.category(don[,-quanti.sup],tab.disj.comp)
    aux <- don
    aux[,-quanti.sup] <- compObs
    compObs <- aux
    
    Tabaux <- cbind.data.frame(tab.disj.comp,don[,quanti.sup,drop=FALSE])
    ordre <- c(which(TabDisjMod!="quanti.sup"),which(TabDisjMod=="quanti.sup"))
    tab.disj.comp <- Tabaux[,order(ordre)]
  }
  
  return(list(tab.disj=tab.disj.comp,completeObs = compObs))
}

library(FactoMineR)
library(missMDA)
data(vnf)
## First the number of components has to be chosen 
##   (for the reconstruction step)
## nb <- estim_ncpMCA(vnf,ncp.max=5) ## Time-consuming, nb = 4

## Impute the indicator matrix and perform a MCA
res.impute <- imputeMCA(vnf, ncp=0)

vnf2 <- vnf[c(1:50),c(1:5)]
tab.disjonctif(vnf2)

for(i in 1:ncol(vnf2)){
  print(unique(vnf2[,i]))
}
data(poison)
tab.disjonctif(poison)
