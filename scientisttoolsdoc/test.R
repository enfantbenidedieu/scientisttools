library(FactoMineR)

# https://openclassrooms.com/fr/courses/5641796-adoptez-visual-studio-comme-environnement-de-developpement/6140631-utilisez-les-raccourcis-pour-etre-plus-efficace

races_canines <- readxl::read_excel("./donnee/races_canines_acm.xlsx")
races_canines <- tibble::column_to_rownames(races_canines,var = "Chien")

# MCA with all informations
res.mca <- MCA(races_canines, ind.sup = 28:33, 
               quanti.sup = 8, quali.sup = 7,  graph=FALSE)
print(res.mca)
res.mca.shiny <- MCAshiny(res.mca)

head(res.mca$ind.sup$coord)

head(res.mca$var$contrib)

# Supplementary variables/categories
head(res.mca$quali.sup$coord)
head(res.mca$quali.sup$cos2)
head(res.mca$quali.sup$v.test)
head(res.mca$quali.sup$eta2)

# Supplementary continuous variables - Coordinates
head(res.mca$quanti.sup$coord)

# Supplementary Individuals
head(res.mca$ind.sup$coord)
head(res.mca$ind.sup$cos2)


summary(res.mca)
dimdesc(res.mca)

res = catdes(tea, num.var=23, proba=0.05)
res$test.chi2
res$category


anova(res.mca$ind$coord[,1]~tea[,1])

library(FactoMineR)
women_work=read.table("http://factominer.free.fr/classical/datasets/women_work.txt", header=TRUE, row.names=1, sep="\t")
res.ca = CA(women_work, col.sup=4:ncol(women_work))

library("FactoMineR")
library(Factoshiny)
library(factoextra)
res.ca <- CA(housetasks, graph = FALSE)
res.shiny = CAshiny(res.ca)
dimdesc(res.ca)
# PCA
data("decathlon2")
decathlon2.active <- decathlon2[1:23, 1:10]
res.pca <- PCA(decathlon2, ind.sup = 24:27, 
               quanti.sup = 11:12, quali.sup = 13, graph=FALSE)

# Supplementary quantitatives informations.
head(res.pca$quanti.sup$coord)
head(res.pca$quanti.sup$cor)
head(res.pca$quanti.sup$cos2)

# Supplementary qualitatives informations
head(res.pca$quali.sup$coord)
head(res.pca$quali.sup$cos2)
head(res.pca$quali.sup$v.test)
head(res.pca$quali.sup$eta2)
head(res.pca$quali.sup$dist)

# Supplementary individuals informations
head(res.pca$ind.sup$coord)
head(res.pca$ind.sup$cos2)
head(res.pca$ind.sup$dist)

p = dimdesc(res.pca)
res.shiny = PCAshiny(res.pca)

model1 <- PCA(decathlon2.active,graph=FALSE)
model2 <- PCA(decathlon2,ind.sup=24:27,quali.sup = 13,graph=FALSE)
model3 <- PCA(decathlon2,quanti.sup = 11:12, quali.sup = 13, graph=FALSE)
model4 <- PCA(decathlon2, ind.sup = 24:27, 
              quanti.sup = 11:12, quali.sup = 13, graph=FALSE)
res.shiny = PCAshiny(model4)


library("FactoMineR")
data(wine)
df <- wine[,c(1,2, 16, 22, 29, 28, 30,31)]
head(df[, 1:7], 4)

library(FactoMineR)
res.famd <- FAMD(df, graph = FALSE)
dimdesc(res.famd)
fviz_contrib(res.famd,choice = "ind")


library(FactoMineR)
data(tea)
res.hcpc = HCPC(res.mca)
res.hcpc$desc.var$test.chi2
res.hcpc$desc.var$category
dimdesc(res.hcpc)

# Multiple Factor Analysis
library(FactoMineR)
library(factoextra)
data(wine)
res.mfa <- MFA(wine, 
               group = c(2, 5, 3, 10, 9, 2), 
               type = c("n", "s", "s", "s", "s", "s"),
               name.group = c("origin","odor","visual",
                              "odor.after.shaking", "taste","overall"),
               num.group.sup = c(1, 6),
               graph = FALSE)

# Quantitative Variables
quanti.var <- get_mfa_var(res.mfa, "quanti.var")
# Coordinates
head(quanti.var$coord)
# Cos2: quality on the factore map
head(quanti.var$cos2)
# Contributions to the dimensions
head(quanti.var$contrib)

# Group of variables
group <- get_mfa_var(res.mfa, "group")
group

head(group$coord)



# Individuals
ind <- get_mfa_ind(res.mfa)
ind
head(ind$coord)
head(ind$coord.partiel)
head(ind$within.inertia)
head(ind$within.partial.inertia)
head(ind$contrib)
head(ind$cos2)

summary(res.mfa)

res.mfa$eig
res.mfa$separate.analyses
res.mfa$group
res.mfa$partial.axes
res.mfa$inertia.ratio
res.mfa$ind
res.mfa$quanti.var
res.mfa$quanti.var.sup
res.mfa$quali.var.sup
res.mfa$summary.quanti
res.mfa$summary.quali
res.mfa.pca <- res.mfa$global.pca








###################### Discriminant Multiple Analysis
library(TExPosition)
race <- read.csv("./donnee/races_canines.txt",sep = "\t",header =TRUE,row.names = )

race <- tibble::column_to_rownames(race,var = "Chien")

canines = list(data=PCAmixdata::tab.disjonctif.NA(race[,c(1:6)]),
               design = PCAmixdata::tab.disjonctif.NA(race[,7]))

disca <- tepDICA(DATA = canines$data,
                 DESIGN = canines$design,
                 make_design_nominal = FALSE,
                 graphs = FALSE)

texdata <- disca$TExPosition.Data
head(texdata$fii) # Factor scores
head(texdata$dii) # Squared distances
head(texdata$rii) # Cosinus of the individuals

head(texdata$eigs)

head(apply(texdata$fii,2,function(x){x^2}))







