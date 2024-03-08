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
head(res.mca$var$eta2)

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

res.mca <- MCA(poison, ind.sup = 53:55, 
               quanti.sup = 1:2, quali.sup = 3:4,  graph=FALSE)
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

X.quanti.sup <- model4$call$quanti.sup
row.w <- model4$call$row.w
dist2 <- as.vector(crossprod(rep(1,nrow(X.quanti.sup)),as.matrix(X.quanti.sup^2*row.w)))

as.matrix(X.quanti.sup^2*row.w)

row_coord <- cbind.data.frame(res.pca$ind$coord[,c(1,2)],
                       Competition=decathlon2[(1:23),13])
df <- row_coord
find_hull <- function(row_coord){
  row_coord[chull(row_coord$Dim.1,row_coord$Dim.2),]
}
hulls <- plyr::ddply(df,"Competition",find_hull)


fig <- ggplot(row_coord,aes(x=Dim.1,y=Dim.2,colour=Competition,fill=Competition))+
  geom_point()+
  geom_polygon(data = hulls,alpha=0.5)


fviz_pca_ind(res.pca,
             habillage = "Competition",
             addEllipses =TRUE, ellipse.type = "confidence",
             palette = "jco", repel = TRUE)


library("FactoMineR")
data(wine)
df <- wine[,c(1,2, 16, 22, 29, 28, 30,31)]
head(df[, 1:7], 4)

library(FactoMineR)
res.famd <- FAMD(df, graph = FALSE)
res.pca <- FAMDshiny(res.famd)
dimdesc(res.famd)
fviz_contrib(res.famd,choice = "ind")

require(FactoMineR)
data(geomorphology)
# FAMD with Factoshiny:
res.shiny=FAMDshiny(geomorphology)

# Find your app the way you left it (by clicking on the "Quit the app" button)
res.shiny2=FAMDshiny(res.shiny)

library(FactoMineR)
data(tea)
res.hcpc = HCPC(res.mca)
res.shiny = HCPCshiny(res.hcpc)

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


res.shiny = MFAshiny(res.mfa)

# Separate analyses
res.mfa$separate.analyses$origin
res.mfa$separate.analyses$odor
res.mfa$separate.analyses$visual
res.mfa$separate.analyses$odor.after.shaking
res.mfa$separate.analyses$taste
res.mfa$separate.analyses$overall

##########################################################
# Individuals Results
##########################################################

# Individuals
ind <- get_mfa_ind(res.mfa)
head(ind$coord)
head(ind$contrib)
head(ind$cos2)
head(ind$coord.partiel)

### à compléter
head(ind$within.inertia)
head(ind$within.partial.inertia)



###########################################################"
# Actives continues variables
#########################################################

# Quantitative Variables
quanti.var <- get_mfa_var(res.mfa, "quanti.var")
# Coordinates
head(quanti.var$coord)
head(quanti.var$cos2)
head(quanti.var$contrib)
head(quanti.var$cor)

#### Supplementary continues variables
quanti.var.sup <- res.mfa$quanti.var.sup
head(quanti.var.sup$coord)
head(quanti.var.sup$cos2)
head(quanti.var.sup$cor)

########################################################
#  Group informations
########################################################

# Group of variables
group <- get_mfa_var(res.mfa, "group")
group$coord
group$contrib
group$Lg
group$RV
group$correlation

# A chercher
group$dist2
group$cos2

### Supp group
group$coord.sup # trouver
group$cos2.sup
group$dist2.sup


summary(res.mfa)

res.mfa$eig
res.mfa$separate.analyses
res.mfa$group$Lg
res.mfa$group$RV

#############################################################
# Partial axes
##############################################################
head(res.mfa$partial.axes$coord)
head(res.mfa$partial.axes$cor)
head(res.mfa$partial.axes$contrib)
dim(res.mfa$partial.axes$cor.between)

fviz_mfa_axes(res2.mfa)

####################" Inertia
res.mfa$inertia.ratio

res.mfa$call$col.w



################################################################
# Supplementary elements
##############################################################"

# Supplementary continue variables
res.mfa$quanti.var.sup$coord
res.mfa$quanti.var.sup$cos2
res.mfa$quanti.var.sup$cor

# Supplementary qualitatives variables
res.mfa$quali.var.sup$coord
res.mfa$quali.var.sup$cos2
res.mfa$quali.var.sup$v.test
res.mfa$quali.var.sup$coord.partiel
# A compléter
res.mfa$quali.var.sup$within.inertia
res.mfa$quali.var.sup$within.partial.inertia

########################## Result summary

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


library("FactoMineR")
data(wine)
df <- wine[,c(1,2, 16, 22, 29, 28, 30,31)]
head(df[, 1:7], 4)
res.famd <- FAMD(df, graph = FALSE)
res.shiny <- FAMDshiny(res.famd)


# Confusion Matrix
library(caret)

# data/code from "2 class example" example courtesy of ?caret::confusionMatrix

lvs <- c("normal", "abnormal")
truth <- factor(rep(lvs, times = c(86, 258)),
                levels = rev(lvs))
pred <- factor(
  c(
    rep(lvs, times = c(54, 32)),
    rep(lvs, times = c(27, 231))),
  levels = rev(lvs))

confusionMatrix(pred, truth)

library(ggplot2)
library(dplyr)

table <- data.frame(confusionMatrix(pred, truth)$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

# fill alpha relative to sensitivity/specificity by proportional outcomes within reference groups (see dplyr code above as well as original confusion matrix for comparison)
ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(good = "green", bad = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))
