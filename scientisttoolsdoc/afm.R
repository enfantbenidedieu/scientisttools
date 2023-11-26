##### Script pour les chapitres 4, 5 et 7 - AFM - Données : jus d'orange (idem chap. 1) 

################################ Chapitre 4 ################################
library(FactoMineR)

orange=read.table("./donnee/Orange5.csv",sep=";",dec=",",header=TRUE,row.names=1)
attributes(orange[,1:17])

# Sélection des variables utilisées en AFM (le même fichier est utilisé ensuite avec plus de variables) 
orange1=orange[,3:18]
attributes (orange1)

ResAFM=MFA(orange1,group=c(8,7,1),type=c("s","s","s"),name.group=c("Chimie","Sensoriel","Appréciation_globale"),num.group.sup=3)

# Graphiques 
plot(ResAFM,choix="ind",hab="group",cex=1.2)
x11()
plot(ResAFM,choix="var",cex=1.2)
x11()
plot(ResAFM,choix="axes")
x11()
plot(ResAFM,choix="group",cex=1.2)

# Ecriture des tableaux de résultats dans un fichier .csv
write.infile(ResAFM,file="ResAFM.csv")

# Diagramme des valeurs propres
barplot(ResAFM$eig[,1],names=1:5)

#### Tableau 4.5

# Initialisation
tab4_5=matrix(nrow=8,ncol=5)

# Noms des lignes et des colonnes
row.names(tab4_5)=c("ACP Chimie","ACP Sensoriel","ACP Ensemble"," Groupe Chimie"," Groupe Sensoriel","AFM"," Groupe Chimie"," Groupe Sensoriel")
colnames(tab4_5)=c("Inertie totale","F1","F2","F1%","F2%")

# Les cinq premiers éléments de la colonne 1
tab4_5[1:5,1]=c(8,7,15,8,7)

# Les lignes (3,4,5), relatives à l'ACP sur les 15 variables, nécessitent d'exécuter cette ACP
resPCA=PCA(orange1[,1:15])

# Ligne 3 : valeurs propres de l'ACP (dans resPCA$eig)
tab4_5[3,2:5]=c(t(resPCA$eig[1:2,1]),t(resPCA$eig[1:2,2]))

# Lignes 4 et 5, colonnes 4 et 5. Les contributions des variables sont dans resPCA$var$contrib ; il faut les additionner par groupe.
tab4_5[4,4:5]=apply(resPCA$var$contrib[1:8,1:2],MARGIN=2,FUN=sum)
tab4_5[5,4:5]=apply(resPCA$var$contrib[9:15,1:2],MARGIN=2,FUN=sum)

# Lignes 4 et 5, colonnes 2 et 3. On retrouve l'inertie en multipliant la contribution (en %) par la valeur propre correspondante.
tab4_5[4:5,2]=tab4_5[4:5,4]*resPCA$eig[1,1]/100
tab4_5[4:5,3]=tab4_5[4:5,5]*resPCA$eig[2,1]/100

# Les ACP séparées des groupes (lignes 1 et 2) sont éditées via l'AFM dans ResAFM$separate.analyses, les valeurs propres étant dans ...$eig.
tab4_5[1,2:5]=c(t(ResAFM$separate.analyses$Chimie$eig[1:2,1]),t(ResAFM$separate.analyses$Chimie$eig[1:2,2]))
tab4_5[2,2:5]=c(t(ResAFM$separate.analyses$Senso$eig[1:2,1]),t(ResAFM$separate.analyses$Senso$eig[1:2,2]))

# Ligne 6. Les valeurs propres de l'AFM sont dans ResAFM$eig.
tab4_5[6,2:5]=c(t(ResAFM$eig[1:2,1]),t(ResAFM$eig[1:2,2]))

# Lignes 6, 7, 8, colonne 1. L'inertie globale d'un groupe dans l'AFM résulte directement du nombre de variables (les variables sont réduites) et de la pondération (par la première valeur propre).
tab4_5[7:8,1]=tab4_5[1:2,1]/tab4_5[1:2,2]
tab4_5[6,1]=tab4_5[7,1]+tab4_5[8,1]

# Lignes 7 et 8. Les inerties des variables sommées par groupe sont dans ResAFM$group ; les inertie brutes sont dans coord (l'explication de ce terme apparaît au chapitre 8) et les pourcentages dans contrib.
tab4_5[7:8,2:3]=c(t(ResAFM$group$coord[,1]),t(ResAFM$group$coord[,2]))
tab4_5[7:8,4:5]=c(t(ResAFM$group$contrib[,1]),t(ResAFM$group$contrib [,2]))

# Pour l'édition, on réduit le nombre de chiffres décimaux.

tab4_5[,2:3]=round(tab4_5[,2:3],3)

tab4_5[,4:5]=round(tab4_5[,4:5],2)

tab4_5[1:5,1]=round(tab4_5[1:5,1],0)

tab4_5[6:8,1]=round(tab4_5[6:8,1],3)

tab4_5
write.infile(tab4_5,file="tab4_5.csv")
#
# Sorties simplifiées éditées dans le fichier SorAFM
summary(ResAFM,nbelements = Inf,file="SorAFM")
#


################################ Chapitre 5 ################################

plot.MFA(ResAFM)

# Obtenir le graphe des individus partiels de l'AFM
# Il est commode de faire apparaître les points partiels non étiquetés mais coloriés
# selon leur groupe (ici on explicite les valeurs par défaut des arguments axes et choix).
plot.MFA(ResAFM,axes=c(1,2),choix="ind",hab="group",partial="all")

# On peut restreindre la représentation des individus partiels à quelques individus
# dont on indique le libellé.
plot.MFA(ResAFM,axes=c(1,2),choix="ind",hab="group",partial=c("P1 Pampryl amb.","P2 Tropicana amb."))

# La sélection interactive des individus dont on souhaite la représentation des points
# partiels (présentée dans R Commander) est accessible directement via la fonction plotMFApartial.

plotMFApartial(ResAFM,axes=c(1,2),hab="group")

# Cette commande affiche la représentation des individus moyens. On clique sur
# les points à sélectionner (un second clic annule la sélection). Attention, toujours
# arrêter explicitement la sélection (clic droit ou onglet en haut à gauche) avant de
# réaliser une autre opération.

# L'étiquetage des points partiels se fait en concaténant le libellé de l'individu et le
# libellé du groupe. Il en résulte un graphique encombré. Exemple :
plot(ResAFM,choix="ind",partial="all",lab.par=T,hab="group")

# Une option consiste à afficher des libellés spécifiques d'un graphique. Pour cela,
# on affiche d'abord un graphique sans libellés ; dans la fenêtre active obtenue, on
# ajoute des libellés à l'aide de la fonction text. Exemple pour un graphique en noir
# et blanc (habillage="none") :
plot(ResAFM,choix="ind",partial="all",hab="none")
text(ResAFM$ind$ coord.partiel[,1],ResAFM$ind$ coord.partiel[,2],rep(c("Chim","Senso"),6),pos=3,offset=0.5 )

# Les sorties de l'AFM comportent un grand nombre de tableaux. Les données du
# tableau 5.1 sont dans resAFM$ind$within.inertia ; celles du tableau 5.2
# sont dans resAFM$inertia.ratio.

####################### Chapitre 7 ####################################

# Graphique de la représentation des groupes
plot.MFA(ResAFM,axes=c(2,3),choix="group")

#### Tableau 7.2 
# Initialisation
tab7_2=matrix(nrow=3,ncol=7)

# Noms des lignes et des colonnes
row.names(tab7_2)=c("W1","W2","NJ")
colnames(tab7_2)=c(paste("Axe",1:5),"Plan(1,2)","Ssp(1,5)")

# Qualité de représentation des groupes
tab7_2[1:2,1:5]=ResAFM$group$cos2[,1:5]

# Inertie projetée de NJ
tab7_2[3,1:5]=apply(ResAFM$group$coord[,1:5]^2,MARGIN=2,FUN=sum)

# Qualité de réprésentation de NJ
# L'inertie totale des Wj est dans ResAFM$group$dist2
tab7_2[3,1:5]=tab7_2[3,1:5]/sum(ResAFM$group$dist2)

# Deux marges colonnes
tab7_2[,6]=apply(tab7_2[1:3,1:2],MARGIN=1,FUN=sum)
tab7_2[,7]=apply(tab7_2[1:3,1:5],MARGIN=1,FUN=sum)

round(tab7_2,4)
#
############# Données : couleur & forme
#
# Lecture et vérification
#
CouleurForme=read.table("Couleur&Forme.csv",header=TRUE,sep=";",dec=".",row.names=1)
CouleurForme
#
# Figure 7.4 (b)
#
res=MFA(CouleurForme,group=c(2,2),type=c("c","c"),graph=F,name.group=c("G1","G2"))
plot(res,choix="group",cex=1.3)
points(1,res$eig[2,1]/res$eig[1,1],pch=16,cex=1.3)
text( 1,res$eig[2,1]/res$eig[1,1],"moyen(AFM)",offset=0.5,pos=3,cex=1.3)

#
############# Données : deux trapèzes (chapitre 5)
#
library(FactoMineR)
trapeze=read.table("DonTrapezes.csv",sep=";",dec=",",header=TRUE,row.names=1)
ResAFM=MFA(trapeze,group=c(2,2),type=c("c","c"),name.group=c("G1","G2"))
#
# Figure 5.7 à gauche
#
coord=rbind(ResAFM$ind$coord[,1:2],ResAFM$ind$coord.partiel[,1:2])
colnames(coord)=c(paste("Dim",1:2,"(",ResAFM$eig[1:2,2],"%)",sep=""))
ylimax= max(coord[,1] ,na.rm=TRUE)+0.5
ylimin= min(coord[,1] ,na.rm=TRUE)-0.5
plot(coord[1:4,1:2],ylim=c(ylimin,ylimax))
text(coord[1:2,1],coord[1:2,2],rownames(coord)[1:2],pos=1,offset=0.5)
text(coord[3:4,1],coord[3:4,2],rownames(coord)[3:4],pos=3,offset=0.5)
abline(v=0)
abline(h=0)
list=c(1:4,1)
points(coord[list,1],coord[list,2],type="o")
#
# Figure 5.7 à droite
#
xlimax=max(coord[,2] ,na.rm=TRUE)+0.5
xlimin=min(coord[,2] ,na.rm=TRUE)-0.5
plot(coord[,1:2],ylim=c(ylimin,ylimax),xlim=c(xlimin,xlimax))
listegauche=c(1,4:6,11,12)
listedroite=c(2:3,7:10)
text(coord[listegauche,1],coord[listegauche,2],rownames(coord)[listegauche],pos=2,offset=0.5)
text(coord[listedroite,1],coord[listedroite,2],rownames(coord)[listedroite],pos=4,offset=0.5)
abline(v=0)
abline(h=0)
list=c(6,8,10,12,6)
points(coord[list,1],coord[list,2],type="o")
list=c(5,7,9,11,5)
points(coord[list,1],coord[list,2],type="o")
#


