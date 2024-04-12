#################################################################
# HCPC on decathlon2
#################################################################

rm(list = ls())
library("FactoMineR")
library(factoextra)

# PCA
res.pca <- PCA(decathlon2, ind.sup = 24:27, 
               quanti.sup = 11:12, quali.sup = 13, graph=FALSE)

res.hc <- HCPC(res.pca,nb.clust = 3,graph = F)

#####

clust <- res.hc$data.clust$clust

# Variables descriptions
desc_var <- res.hc$desc.var
desc_var$quanti.var
desc_var$quanti
desc_var$call

#### Axis description
desc_axes <- res.hc$desc.axes
desc_axes$quanti.var
desc_axes$quanti
desc_axes$call

ind <- res.hc$desc.ind
ind$para
ind$dist

consolidation <- function(X, clust, iter.max = 10, ...) {
  centers <- NULL
  centers <- by(X, clust, colMeans)
  centers <- matrix(unlist(centers), ncol = ncol(X), byrow = TRUE)
  km <- kmeans(X, centers = centers, iter.max = iter.max, ...)
  return(km)
}


X <- res.pca$ind$coord
clust <- res.hc$data.clust$clust
cons <- consolidation(X,clust)

cons$cluster
cons$centers
cons$size
cons$totss
cons$withinss
cons$betweenss

sum(cons$withinss)

km <- kmeans(X)

url = "D:/Bureau/PythonProject/packages/scientisttools/scientisttools/data"
data <- readxl::read_excel(paste0(url,"/temperature.xlsx"))
data <- data %>% tibble::column_to_rownames(var="Villes")

res.pca = PCA(data)
res.hc <- HCPC(res.pca,nb.clust = 3,graph=F)

fviz_cluster(res.hc)

res.hc$desc.var$quanti.var
res.hc$desc.var$quanti

res.hc$desc.ind$para
