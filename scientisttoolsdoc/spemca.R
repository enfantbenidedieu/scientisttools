library(GDAtools)
data(Music)
getindexcat(Music[,1:5])
mca <- speMCA(Music[,1:5],excl=c(3,6,9,12,15))
#
mca$eig
attributes(mca)

head(mca$ind$coord)
head(mca$ind$contrib)
head(mca$ind$cos2)

head(mca$var$coord)

data(Music)
junk <- c("FrenchPop.NA", "Rap.NA", "Rock.NA", "Jazz.NA", "Classical.NA")
mca <- speMCA(Music[,1:5], excl = junk)
# This is equivalent to :
mca <- speMCA(Music[,1:5], excl = c(3,6,9,12,15))

mca$call$marge.col


library(FactoMineR)

music <- dplyr::bind_rows(Music, Music[,1:5])
rownames(music) <- 1:nrow(music)
res.mca <- MCA(music, excl = c(3,6,9,12,15),graph = F,quali.sup = c(6:ncol(music)),
               ind.sup = c(501:1000))

res <- FactoMineR::predict.MCA(res.mca,Music[,1:5])
res.mca$var$eta2
