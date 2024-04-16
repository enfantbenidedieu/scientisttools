#################################################################
# AFM sur Données Mixtes
#################################################################

#### Script pour le chapitre 8 - AFM sur données mixtes - Données : biométrie2
rm(list = ls())
library(FactoMineR)

Biometrie2=read.table("donnee/Biometrie2.csv",header=TRUE,sep=";",dec=".",row.names=1)

colnames(Biometrie2)

# L'AFM est réalisée sur les seules six premières colonnes avec toutes les options par défaut

res=MFA(Biometrie2,group=c(3,3,3,3),
        type=c("n","s","n","s"),
        #num.group.sup = c(3,4),
        graph = FALSE)

fviz_contrib(res,choice = "quanti.var")

fviz_mfa_ind(res)
fviz_mfa_var(res)
fviz_mfa_axes(res)

####################################################
# Eigenvalues informations
########################################################
res$eig

#######################################################
# Individuals informations
#######################################################
res$ind$coord
res$ind$cos2
res$ind$contrib
res$ind$coord.partiel
res$ind$within.inertia
res$ind$within.partial.inertia

#########################################################
# Variables Informations
##########################################################
quanti_var <- get_mfa_var(res,"quanti.var")
quanti_var$coord
quanti_var$cor
quanti_var$cos2
quanti_var$contrib

###############################################################
# Qualitatives informations
###############################################################
quali_var <- get_mfa_var(res,"quali.var")
quali_var$coord
quali_var$cos2
quali_var$contrib
quali_var$v.test
quali_var$coord.partiel
quali_var$within.inertia
quali_var$within.partial.inertia

#############################################################
# Group informations
###############################################################
group <- get_mfa_var(res,"group")
head(group$coord)
head(group$contrib)
head(group$correlation)
head(group$Lg)
head(group$RV)
head(group$dist2) 
head(group$cos2)

x <- group$Lg[c(1,2),c(1,2)]

res$group$coord.sup
res$group$cos2.sup
res$group$dist2.sup

#############################################################
# Partial axes
##############################################################
partial_axes <- get_mfa_partial_axes(res)
partial_axes$coord
partial_axes$cor
partial_axes$contrib
partial_axes$cor.between

################################################################
# Inertia Informations
################################################################
res$inertia.ratio

# Figure 8.1 à droite
varco<-res$quali.var$coord
plot.MFA(res,choix="ind",hab="group")
points(varco[1:3,1],varco[1:3,2],type="o")
points(varco[6:8,1],varco[6:8,2],type="o")

# Etiqueter les individus partiels et/ou les modalités partielles, avec des noms de groupes choisis :
res=MFA(Biometrie2[,1:6],group=c(3,3),type=c("n","s"),name.group=c("Classes","CR"))
plot.MFA(res,axes=c(1,2),choix="ind",habillage="group",invisible="quali",partial="all",lab.par=TRUE)

### Graphe des modalités - figure 8.2
plot.MFA(res,axes=c(1,2),choix="ind",habillage="group",invisible="ind",partial="all")
# Etiqueter les points partiels par les noms des groupes
text(res$quali.var$coord.partiel[seq(1,15,2),1],res$quali.var$coord.partiel[seq(1,15,2),2],rep("CR",6),pos=3,offset=0.5)
text(res$quali.var$coord.partiel[seq(2,16,2),1],res$quali.var$coord.partiel[seq(2,16,2),2],rep("Classes",6),pos=1,offset=0.5)
# Relier les points moyens des modalités des variables Longueur et Largeur
points(varco[1:3,1],varco[1:3,2],type="o")
points(varco[6:8,1],varco[6:8,2],type="o")

### Colorier les individus en fonction de leur modalité pour la variable 2
plot.MFA(res,choix="ind",invisible="quali",hab=2)

### Afficher des ellipses de confiance autour des modalités de la variable 2
plotellipses(res,keepvar=2)

### Figure 8.3
# Individus et modalités sur plan 1 3
plot.MFA(res,choix="ind",axes=c(1,3),hab="group")
points(varco[1:3,1],varco[1:3,3],type="o")
points(varco[6:8,1],varco[6:8,3],type="o")

# Carré des liaisons 
res=MFA(Biometrie2,group=c(3,3,rep(1,6)),type=c("n","s",rep("n",3),rep("s",3)),num.group.sup=c(3:8),name.group=c("G1qualitatif","G2quantitatif","LongQuali","PoidQuali","LargQuali","LongQuanti","PoidQuanti","LargQuanti"),graph = F)

### Tableau 8.5 (corrélations entre axes partiels)
round(res$partial.axes$cor.between[6:8,1:3],2)

### Tableau 8.8 (qualités de représentation dans RI2 ; avec des libellés un peu abrégés)
# Initialisation et choix des libellés
tab_8.8=matrix(nrow=9,ncol=3)
row.names(tab_8.8)=c("Groupe 1","Groupe 2","Ensemble","LongQuali","PoidQuali","LargQuali","LongQuanti","PoidQuanti","LargQuanti")
colnames(tab_8.8)=c("F1","F2","F3")
# Les cos2 des Wj sont calculés par l'AFM (cos2 et cos2.sup).
tab_8.8[1:2,1:3]=res$group$cos2[,1:3]
tab_8.8[4:9,1:3]=res$group$cos2.sup[,1:3]
# Le rapport (pour NJ) inertie projetée/inertie totale doit être recalculé à partir des coordonnées (coord)
# et des distances des Wj à l'origine (dist2)
tab_8.8[3,1:3]= apply(res$group$coord[,1:3]^2,MARGIN=2,FUN=sum)
tab_8.8[3,1:3]=tab_8.8[3,1:3]/sum(res$group$dist2)
# Edition avec deux chiffres décimaux
round(tab_8.8,2)

### Tableau 8.9 (indicateurs de liaison Lg et RV )
# Initialisation et choix des libellés
tab8_9=matrix(nrow=2,ncol=3)
row.names(tab8_9)=c("Lg","RV")
colnames(tab8_9)=c("Longueur","Poids","Largeur")
# On récupère les coefficients sur les diagonales # des matrices res$group$Lg et res$group$RV
tab8_9[1,1:3]<-diag(res$group$Lg[3:5,6:8] )
tab8_9[2,1:3]<- diag(res$group$RV[3:5,6:8] )
# Edition avec trois chiffres décimaux
round(tab8_9,3)

### Tableau 8.7
# Initialisation
Tab8_7=matrix(rep(0,78),nrow=13,ncol=6)
# BCR Groupe quantitatif centré et réduit
# Bdis : groupe qualitatif sous forme disjonctive complète
BCR=as.matrix(scale(Biometrie2[,4:6])*sqrt(6/5))
Bdis=tab.disjonctif(Biometrie2[,1:3])
colnames(Tab8_7)=rownames(Biometrie2)
rownames(Tab8_7)=c(colnames(BCR),"Ind.part.quanti",colnames(Bdis),"Ind.part.quali")

# Premières valeurs propres de l'AFM et des analyses séparées
L1AFM=res$eig[1,1]
L1ACM=res$separate.analyses$G1qualitatif$eig[1,1]
L1ACP=res$separate.analyses$G2quantitatif$eig[1,1]

# coeff=coefficient dans la relation de transition
coord=res$quanti.var$coord[,1]
coeff=2/(sqrt(L1AFM)*L1ACP)

# Partie groupe quantitatif
for(i in 1:3){ for(j in 1:6) {
  Tab8_7[i,j]=BCR[j,i]*coord[i]*coeff
  Tab8_7[4,j]=Tab8_7[4,j]+Tab8_7[i,j]
}}

# Partie groupe qualitatif
coeff=2/(L1AFM*L1ACM*3)
coord=res$quali.var$coord[,1]
for(i in 1:8){for(j in 1:6) {
  Tab8_7[i+4,j]=Bdis[j,i]*coord[i]*coeff
  Tab8_7[13,j]=Tab8_7[13,j]+Tab8_7[i+4,j]
}}
