
library(FactoMineR)
data(tea)
res.mca = MCA(tea, ncp=20, quanti.sup=19, quali.sup=c(20:36), graph=FALSE)

res.hcpc = HCPC(res.mca,graph = F)


table(res.hcpc$data.clust$clust)

res.hcpc$desc.var$test.chi2

res.hcpc$desc.var$category$`1`
res.hcpc$desc.var$category$`2`

res.hcpc$desc.axes$quanti.var



aux4 <- min(phyper(round(Table[j,k],0)-1,round(marge.li[j],0),round(sum(marge.li),0)-round(marge.li[j],0),round(marge.col[k],0))*2+dhyper(round(Table[j,k],0),round(marge.li[j],0),round(sum(marge.li),0)-round(marge.li[j],0),round(marge.col[k],0)),
            phyper(round(Table[j,k],0),round(marge.li[j],0),round(sum(marge.li),0)-round(marge.li[j],0),round(marge.col[k],0),lower.tail=FALSE)*2+dhyper(round(Table[j,k],0),round(marge.li[j],0),round(sum(marge.li),0)-round(marge.li[j],0),round(marge.col[k],0)))