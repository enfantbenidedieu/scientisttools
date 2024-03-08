################################################################################
# CA with children dataset
################################################################################

# Multiple Factor Analysis
rm(list = ls())
library(FactoMineR)
library(factoextra)

data(children)
children$group <- c(rep("A",4),rep("B",5),rep("C",5),rep("D",4))
res.ca <- CA (children, row.sup = 15:18, quanti.sup = 6:8,quali.sup = 9,graph = F)

############"" Eigenvalyes
res.ca$eig


############# Rows informations
rows = get_ca_row(res.ca)
head(rows$coord)
head(rows$contrib)
head(rows$cos2)
head(rows$inertia)


####################### Columns informations
cols = get_ca_col(res.ca)
head(cols$coord)
head(cols$contrib)
head(cols$cos2)
head(cols$inertia)


res.km <- kmeans(cols$coord, centers = 2, nstart = 25)
grp <- as.factor(res.km$cluster)
# Color variables by groups
fviz_ca_col(res.ca, col.var = grp, 
             palette = c("#0073C2FF", "#EFC000FF"),
             legend.title = "Cluster")
