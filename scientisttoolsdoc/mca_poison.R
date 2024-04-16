rm(list = ls())
library(FactoMineR)
library(factoextra)
data (poison)
res.mca <- MCA(poison,ind.sup = c(51:55),quanti.sup = c(1:2),quali.sup = c(3:4),graph = FALSE)


fviz_cos2(res.mca,choice = "quanti.sup")

res.mca$call$marge.col
res.mca$call$marge.row
res.mca$svd$vs
res.mca$svd$U
res.mca$svd$V

head(res.mca$ind$coord)

res.mca$quali.sup$eta2

res.mca$quanti.sup$coord

head(res.mca$var$coord)
unique(poison$Sick)
unique(poison$Sex)


data(geomorphology)
res <- FAMD(geomorphology,graph = FALSE)

head(res$quanti.var$coord)
head(res$var$coord)




get_mca_var(res.mca)
x = get_mca(res.mca,element = "quanti.sup")
x = get_mca_var(res.mca,element = "quanti.sup")
x = get_mca_var(res.mca,element = "mca.cor")


fviz_mca_var(res.mca,choice = "quanti.sup")

###############################################
# HCPC with poison
####################################################"
res.hc <- HCPC(res.mca,graph = F)

desc_var <- res.hc$desc.var
desc_var$test.chi2
desc_var$category
desc_var$quanti.var
desc_var$quanti


X <- res.mca$ind$coord
clust <- res.hc$data.clust$clust

centers <- by(X, clust, colMeans)
centers <- matrix(unlist(centers), ncol = ncol(X), byrow = TRUE)

df <- cbind.data.frame(X,clust)

df %>% group_by(clust) %>% 
  summarise(across(
    .cols = is.numeric, 
    .fns = mean, na.rm = TRUE
  ))

library(ggplot2)
library(ggdendro)
model <- hclust(dist(USArrests), "ave")
dhc <- as.dendrogram(model)
# Rectangular lines
ddata <- dendro_data(dhc, type = "rectangle")

p <- ggplot(segment(ddata)) + 
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend)) + 
  coord_flip() + 
  scale_y_reverse(expand = c(0.2, 0))
p