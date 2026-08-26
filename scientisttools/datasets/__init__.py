# -*- coding: utf-8 -*-
from __future__ import annotations
import pathlib
from pandas import DataFrame, read_excel, read_csv
from pyreadr import read_r
from collections import namedtuple

# https://husson.github.io/data.html
# https://r-stat-sc-donnees.github.io/liste_don.html

def namedtupledocstring(docstring, *ntargs):
    nt = namedtuple(*ntargs)
    class Dataset(nt):
        __doc__ = docstring
    return Dataset

DATASETS_DIR = pathlib.Path(__file__).parent / "data"

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Datasets as DataFrame
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------ ardeche dataset ----------------------------------------------------- 
ardeche = {
    "data" : read_excel(DATASETS_DIR/"ardeche.xlsx",index_col=0,header=0),
    "col_group" : (5,6,6,6,6,6),
    "row_group" : (11,3,13,16),
    "name_col_group" : ("Jul82","Aug82","Nov82","Feb83","Apr83","Jul83"),
    "name_row_group" : ("Ephemeroptera","Plecoptera","Coleoptera","Trichoptera")
}
__doc__ = """
Ardeche Dataset

Faua table with double (row and column) partitioning.
This data set gives information about species of benthic macroinvertebrates in different sites and dates.

Returns
-------   
ardeche : Dataset
    An object with the following attributes:

    data : DataFrame of shape (43,35)
        Fauna Table with double (row and column) partitioning : 43 species (rows) and 35 samples (columns)
    col_group : tuple, default = (5,6,6,6,6,6)
        The number of columns in each column group. Its containing the repartition of samples for the 6 dates. july 1982, august 1982, november 1982, february 1983, april 1983 and july 1983.
    row_group : tuple, default = (11,3,13,16)
        The number of rows in each row group. Its containing the repartition of species in the 4 groups defining the species order. Ephemeroptera, Plecoptera, Coleoptera, Trichoptera
    name_col_group : tuple, default = ("Jul82","Aug82","Nov82","Feb83","Apr83","Jul83")
        The name of the columns groups. 
    name_col_group : tuple, default = ("Ephemeroptera","Plecoptera","Coleoptera","Trichoptera")
        The name of the rows groups.

References
----------
[1] Cazes, P.; Chessel, D.; Doledec, S. L'analyse des correspondances internes d'un tableau partitionné : son usage en hydrobiologie. Revue de Statistique Appliquée, Volume 36 (1988) no. 1, pp. 39-54. https://www.numdam.org/item/RSA_1988__36_1_39_0/

Examples
--------
>>> from scientisttools.datasets import ardeche
>>> from scientisttools import CA, ICA
>>> #correspondence analysis (CA)
>>> clf = CA()
>>> clf.fit(ardeche.data)
CA()
>>> #internal correspondence analysis (ICA)
>>> clf = ICA(row_group=ardeche.row_group, name_row_group=ardeche.name_row_group, col_group=ardeche.col_group, name_col_group=ardeche.name_col_group)
>>> clf.fit(ardeche.data)
ICA(row_group=(11,3,13,16),name_row_group=("Ephemeroptera","Plecoptera","Coleoptera","Trichoptera"),col_group=(5,6,6,6,6,6),name_col_group=("Jul82","Aug82","Nov82","Feb83","Apr83","Jul83"))
"""
ardeche = namedtupledocstring(__doc__,"ardeche",ardeche.keys())(*ardeche.values())

#------------------------------------------ autos1990 dataset ----------------------------------------------------- 
autos1990 = read_csv(DATASETS_DIR/'autos1990.txt',delimiter=" ",header=0,index_col=0)
autos1990.__doc__ = """
Autos 1990 Dataset

A data with 27 individuals and 9 variables (6 quantitative and 3 qualitative).

References
----------
[1] Abdesselam, R. (2006), `Analyse en Composantes Principales Mixte <https://perso.univ-lyon2.fr/~rabdesse/fr/Publications/RNTI.pdf>`_, Revue Nouvelles Technologies Information.

Examples
--------
>>> from scientisttools.datasets import autos1990
>>> from scientisttools import MPCA
>>> clf = MPCA()
>>> clf.fit(autos1990)
MPCA()
"""

#------------------------------------------ autos2005 dataset ----------------------------------------------------- 
autos2005 = {
    "actif" : read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "ind_sup" : read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "sup_var": read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil3",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil4",index_col=0,header=0)
}
__doc__ = """
Autos 2005 Dataset

Returns
-------
autos2005 : Dataset
    An object with the following attributes:

    actif: DataFrame of shape (38, 12)
        Actifs dataset.
    ind_sup: DataFrame of shape (7, 12)
        Supplementary individuals dataset.
    sup_var: DataFrame of shape (38, 3)
        Supplementary variables dataset.
    data: DataFrame of shape (45, 15)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import autos2005
>>> from scientisttools import FAMD
>>> clf = FAMD(ind_sup=range(38,45),sup_var=range(12,16))
>>> clf.fit(autos2005.data)
FAMD(ind_sup=range(38,45),sup_var=range(12,16))
"""
autos2005 = namedtupledocstring(__doc__,"autos2005",autos2005.keys())(*autos2005.values())

#------------------------------------------ autos2006 dataset ----------------------------------------------------- 
autos2006 = {
    "actif" : read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "ind_sup" : read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil3",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil4",index_col=0,header=0)
}
__doc__ = """
Autos 2006 Dataset

The dataset contains 20 autos.

Returns
-------
autos2006 : Dataset
    An object with the following attributes:

    actif: DataFrame of shape (18,6)
        Input data for actifs elements.
    ind_sup: DataFrame of shape (2,6)
        for supplementary individuals dataset.
    sup_var: DataFrame of shape (18,3)
        Input data for supplementary variables.
    data: DataFrame of shape (20,9)
        Overall dataset.

References
----------
[1] Saporta G. (2011), « `Probabilites, Analyse des données et Statistiques <https://www.editionstechnip.com/en/catalogue-detail/149/probabilites-analyse-des-donnees-et-statistique.html>`_», Editions TECHNIP, 3ed.

[2] Rakotomalala R. (2020), Pratique des méthodes factorielles avec Python, Version 1.0, Université Lumière Lyon 2. https://hal.science/hal-04868625v1

Examples
--------
>>> from scientisttools.datasets import autos2006
>>> from scientisttools import PCA
>>> clf = PCA(partial=0,ind_sup=(18,19),sup_var=(6,7,8))
>>> clf.fit(autos2006.data)
PCA(ind_sup=(18,19),partial=0,sup_var=(6,7,8))
"""
autos2006 = namedtupledocstring(__doc__, "autos2006",autos2006.keys())(*autos2006.values())

#------------------------------------------ autosmds dataset ----------------------------------------------------- 
autosmds = {
    "actif" : read_excel(DATASETS_DIR/"autosmds.xlsx",sheet_name="Sheet1",index_col=0,header=0),
    "ind_sup" : read_excel(DATASETS_DIR/"autosmds.xlsx",sheet_name="Sheet2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"autosmds.xlsx",sheet_name="Sheet3",index_col=0,header=0)
}
__doc__ = """
Autos Multidimensional Scaling Dataset

Returns
-------
autosmds : Dataset
    An object with the following attributes:

    actif: DataFrame of shape (18,6)
        Input data for actifs elements.
    ind_sup: DataFrame of shape (2,6)
        for supplementary individuals dataset.
    data: DataFrame of shape (20,9)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import autosmds
>>> from scientisttools import PCoA
>>> clf = PCoA(ncp=2,metric="euclidean",ind_sup=(12,13,14))
>>> clf.fit(autosmds)
PCoA(ind_sup=(12,13,14),ncp=2)
"""
autosmds = namedtupledocstring(__doc__,"autosmds",autosmds.keys())(*autosmds.values())

#------------------------------------------ autosmds2 dataset ----------------------------------------------------- 
autosmds2 = {
    "actif" : read_excel(DATASETS_DIR/"autosmds.xlsx",sheet_name="Sheet4",index_col=0,header=0),
    "ind_sup" : read_excel(DATASETS_DIR/"autosmds.xlsx",sheet_name="Sheet5",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"autosmds.xlsx",sheet_name="Sheet6",index_col=0,header=0)
}
__doc__ = """
Autos Multidimensional Scaling Dataset

Returns
-------
autosmds2 : Dataset
    An object with the following attributes:

    actif: DataFrame of shape (18,6)
        Input data for actifs elements.
    ind_sup: DataFrame of shape (2,6)
        for supplementary individuals dataset.
    data: DataFrame of shape (20,9)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import autosmds2
>>> from scientisttools import PCoA
>>> clf = PCoA(ncp=2,metric="precomputed",ind_sup=(12,13,14))
>>> clf.fit(autosmds2)
PCoA(ind_sup=(12,13,14),ncp=2)
"""
autosmds2 = namedtupledocstring(__doc__,"autosmds2",autosmds2.keys())(*autosmds2.values())

#------------------------------------------ beer dataset ----------------------------------------------------- 
beer = read_excel(DATASETS_DIR/"beer_rnd.xlsx",index_col=None,header=0)
beer.__doc__ = """
Beer Dataset

Reference
---------
[1] Rakotomalala R, https://eric.univ-lyon2.fr/ricco/tanagra/fichiers/fr_Tanagra_Principal_Factor_Analysis.pdf

Examples
--------
>>> from scientisttools.datasets import beer
>>> from scientisttools import FA
>>> clf = FA()
>>> clf.fit(beer)
FA()
"""

#------------------------------------------ body dataset ----------------------------------------------------- 
body = read_excel(DATASETS_DIR/"body.xlsx",sheet_name="body",index_col=None,header=0)
body.__doc__ = """
Body Dimensions Datasets

The data give some body dimension measurements as well as age, weight, height, and gender on 507 individuals. 
The 247 men and 260 women were primarily individuals in their twenties and thirties, with a scattering of older men and women, all exercising serveral hours a week. 
A data with 507 observations and 15 variables:

    * shoulder.girth: shoulder girth (in cm) -- épaule (fr)
    * chest.girth: Chest girth (in cm) -- poitrine (fr)
    * waist.girth: Waist girth (in cm) -- taille (fr)
    * navel.girth: Navel girth (in cm) -- nombril (fr)
    * hip.girth: Hip girth (in cm) -- hanche (fr)
    * thigh.girth: Thigh girth (in cm) -- cuisse (fr)
    * bicep.girth: Bicep girth (in cm) -- biceps (fr)
    * forearm.girth: Forearm girth (in cm) -- avant-bras (fr)
    * knee.girth: Knee girth (in cm) -- genou (fr)
    * calf.girth: Calf girth (in cm) -- mollet (fr)
    * ankle.girth: Ankle girth (in cm) -- cheville (fr)
    * wrist.girth: Wrist girth (in cm) -- poignet (fr)
    * weight: Weight (in kg)
    * height: Height (in cm)
    * gender: Gender ; 1 for males and 0 for females.

Examples
--------
>>> from scientisttools.datasets import body
>>> from scientisttools import PCA
>>> clf = PCA(partial=(12,13,14))
>>> clf.fit(body)
"""

#------------------------------------------ burger dataset ----------------------------------------------------- 
burger = read_excel(DATASETS_DIR/"burger.xlsx",sheet_name="Feuil1",index_col=0,header=0)
burger.__doc__ = """
Burger King Dataset

Examples
--------
>>> from scientisttools.datasets import burger
>>> from scientisttools import PCA
>>> clf = PCA(ncp=3,sup_var=(10,11))
>>> clf.fit(burger)
"""

#------------------------------------------ burgundywines dataset ----------------------------------------------------- 
data = DataFrame(
        data=[
            [1, 6, 7, 2, 5, 7, 6, 3, 6, 7],
            [5, 3, 2, 4, 4, 4, 2, 4, 4, 3],
            [6, 1, 1, 5, 2, 1, 1, 7, 1, 1],
            [7, 1, 2, 7, 2, 1, 2, 2, 2, 2],
            [2, 5, 4, 3, 5, 6, 5, 2, 6, 6],
            [3, 4, 4, 3, 5, 4, 5, 1, 7, 5],
        ],
        columns= ["Fruity one","Woody one","Coffee","Red fruit","Roasted","Vanillin","Woody two","Fruity three","Butter","Woody three"],
        index=[f"Wine{i+1}" for i in range(6)],
    )
data.insert(0, "Oak type", [1, 2, 2, 2, 1, 1])
burgundywines = {
    "data" : data,
    "group" : (1,3,4,3),
    "name" : ("type","expert1","expert2","expert3") 
}
__doc__ = """
Burgundy Wines Dataset

Returns
-------
burgundywines : Dataset
    An object with the following attributes:

    data: DataFrame of shape (6,11)
        Overall dataset.
    group: tuple
        Number of variables in each group.
    name: list
        Name of the groups.

References
----------
[1] https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf

[2] https://personal.utdallas.edu/~herve/Abdi-MFA2007-pretty.pdf

Examples
--------
>>> from scientisttools.datasets import burgundywines
>>> from scientisttools import MFA
>>> clf = MFA(group=burgundywines.group,name_group=burgundywines.name,group_type=("s","s","s","s"),num_group_sup=0)
>>> clf.fit(wines)
MFA(group=(1,3,4,3),name_group=("type","expert1","expert2","expert3"),group_type=("s","s","s","s"),num_group_sup=0)
"""

#------------------------------------------ canines dataset ----------------------------------------------------- 
canines = {
    "actif" : read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil1",header=0,index_col=0),
    "ind_sup" : read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil2",header=0,index_col=0),
    "sup_var" : read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil3",header=0,index_col=0),
    "data" : read_excel(DATASETS_DIR/"canines.xlsx",sheet_name="Feuil4",header=0,index_col=0)
}
__doc__ = """
Canines Dataset

The data contains 32 individuals

Returns
-------
canines : Dataset
    An object with the following attributes:

    actif: DataFrame of shape (27,6)
        Input data for actifs elements.
    ind_sup: DataFrame of shape (5,6)
        Input data for supplementary individuals dataset.
    sup_var: DataFrame of shape (27, 2)
        Input data for supplementary variables.
    data: DataFrame of shape (32,8)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import canines
>>> from scientisttools import MCA
>>> clf = MCA(ind_sup=range(27,32),sup_var=(6,7))
>>> clf.fit(canines.data)
MCA(ind_sup=(27,28,29,30,31),sup_var=(6,7))
"""
canines = namedtupledocstring(__doc__,"canines",canines.keys())(*canines.values())

#------------------------------------------ children dataset ----------------------------------------------------- 
children = {
    "actif" : read_excel(DATASETS_DIR/"children.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "row_sup" : read_excel(DATASETS_DIR/"children.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "col_sup" : read_excel(DATASETS_DIR/"children.xlsx",sheet_name="Feuil3",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"children.xlsx",sheet_name="Feuil4",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"children.xlsx",sheet_name="Feuil5",index_col=0,header=0)
}
__doc__ = """
Children dataset

The data used here is a contingency table that summarizes the answers given by different categories of people to the following question : 
according to you, what are the reasons that can make hesitate a woman or a couple to have children?
A data frame with 18 rows and 9 columns. Rows represent the different reasons mentioned, columns represent the different categories (education, age) people belong to.

Returns
-------
children : Dataset
    An object with the following attributes:

    actif: DataFrame of shape (14, 5)
        Input data for actifs elements.
    row_sup: DataFrame of shape (4, 5)
        Input data for supplementary individuals.
    col_sup: DataFrame of shape (14, 3)
        Input data for supplementary columns.
    sup_var: DataFrame of shape (14, 3)
        Input data for supplementary variables.
    data: DataFrame of shape (18, 8)
        Overall dataset

Examples
--------
>>> from scientisttools.datasets import children
>>> #with supplementary columns and supplementary qualitative
>>> clf = CA(row_sup=range(14,18),col_sup=(5,6,7),sup_var=8)
>>> clf.fit(children.data)
CA(col_sup=(5,6,7),row_sup=range(14,18),sup_var=8)
>>> #with supplementary quantitative and supplementary qualitative
>>> clf2 = CA(row_sup=range(14,18),sup_var=range(5,9))
>>> clf2.fit(children.data)
CA(row_sup=range(14,18),sup_var=range(5,9))
"""
children = namedtupledocstring(__doc__,"children",children.keys())(*children.values())

#------------------------------------------ cultural dataset ----------------------------------------------------- 
cultural = read_csv(DATASETS_DIR/'cultural.txt',delimiter=",",header=0,index_col=0)
cultural.__doc__ = """
Cultural Dataset

Examples
--------
>>> from scientisttools.datasets import cultural
>>> from scientisttools import CA
>>> clf = CA(group=0)
>>> clf.fit(autos1990)
CA(group=0)
"""

#------------------------------------------ decathlon dataset ----------------------------------------------------- 
decathlon = {
    "actif" : read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "ind_sup" : read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil3",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil4",index_col=0,header=0)
}
__doc__ = """
Performance in decathlon (data)

The data used here refer to athletes' performance during two sporting events.
A data with 46 rows and 13 columns: the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. 
The columns 11 and 12 correspond respectively to the rank and the points obtained. 
The last column is a categorical variable corresponding to the sporting event (2004 Olympic Game or 2004 Decastar)
Supplementary individuals are the top 5 from the 1988 Seoul Olympics.

A data with 46 rows and 13 columns: 

    * the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. 
    * The columns 11 and 12 correspond respectively to the rank and the points obtained. 
    * The last column is a categorical variable corresponding to the sporting event (2004 Olympic Game or 2004 Decastar)
    * Supplementary individuals are the top 5 from the 1988 Seoul Olympics.

Returns
-------
decathlon : decathlon
    An object with the following attributes:

    actif: DataFrame of shape (41,10)
        Input data for actifs elements.
    ind_sup: DataFrame of shape (5,10)
        Input data for supplementary individuals dataset.
    sup_var: DataFrame of shape (41, 2)
        Input data for supplementary variables.
    data: DataFrame of shape (46,12)
        Overall dataset

Examples
--------
>>> from scientisttools.datasets import decathlon
>>> from scientisttools import PCA
>>> clf = PCA(ind_sup=range(41,46),sup_var=(10,11,12))
>>> clf.fit(decathlon.data)
PCA(ind_sup=range(41,46),sup_var=(10,11,12))
"""
decathlon = namedtupledocstring(__doc__,"decathlon",decathlon.keys())(*decathlon.values())

#------------------------------------------ decathlon2 dataset ----------------------------------------------------- 
decathlon2 = {
    "actif" : read_excel(DATASETS_DIR/"decathlon2.xlsx",sheet_name="Sheet1",index_col=0,header=0),
    "ind_sup" : read_excel(DATASETS_DIR/"decathlon2.xlsx",sheet_name="Sheet2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"decathlon2.xlsx",sheet_name="Sheet3",index_col=0,header=0),
    "group" : (4,6),
    "name" : ("Speed","Strenght")
}
__doc__ = """
Performance in decathlon (data)

The data used here refer to athletes' performance during two sporting events.
A data with 46 rows and 10 columns: the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. 
Supplementary individuals are the top 5 from the 1988 Seoul Olympics.

Returns
-------
decathon2 : Dataset
    An object with the following attributes:

    actif: DataFrame of shape (41,10)
        Input data for actifs elements.
    ind_sup: DataFrame of shape (5,10)
        Input data for supplementary individuals dataset.
    data: DataFrame of shape (46,10)
        Overall dataset
    group : tuple, default = (4,6)
        The number of variables in each group
    name : tuple, default = ("Speed","Strenght")
        The name of the groups.

Examples
--------
>>> from scientisttools.datasets import decathlon2
>>> from scientisttools import DISTATIS
>>> clf = DISTATIS(group=decathlon2.group,name_group=decatlon2.name,ind_sup=range(41,46))
>>> clf.fit(decathlon.data)
DISTATIS(group=(4,6),ind_sup=range(41,46),name_group=("Speed","Strenght"))
"""
decathlon2 = namedtupledocstring(__doc__,"decathlon2",decathlon2.keys())(*decathlon2.values())

#------------------------------------------ distalgo dataset ----------------------------------------------------- 
distalgo = {
    "data" : read_excel(DATASETS_DIR/"distalgo.xlsx",index_col=0,header=0),
    "group" : (6,6,6,6),
    "name" : ("Pixels","Measures","Ratings","Pairwise") 
}
__doc__ = """
Four computer algorithms evaluate the similarity of six faces for DISTATIS analysis

This dataset is used to illustrate the use of DISTATIS. Four algorithms (Pixels, Measures, Ratings and Pairwise) evaluate the similarity (i.e., distance) between six faces (3 females and 3 males). 
Each algorithm provides a :math:6 \times 6 distance matrix evaluating the distance between each pair of faces.

Returns
-------   
distalgo : Dataset
    An object with the following attributes:
    
    data : DataFrame of shape (6,24)
        Overall dataset.
    group : tuple, default = (6,6,6,6)
        The number of columns in each group. 
    name : tuple, default = ("Pixels","Measures","Ratings","Pairwise")
        The name of the columns groups.

References
----------
[1] Abdi, H., Valentin, D., O'Toole, A.J., & Edelman, B. (2005). DISTATIS: The analysis of multiple distance matrices. Proceedings of the IEEE Computer Society: International Conference on Computer Vision and Pattern Recognition. (San Diego, CA, USA). pp. 42–47.

Examples
--------
>>> from scientisttools.datasets import distalgo
>>> from scientisttools import DISTATIS
>>> clf = DISTATIS(group=distalgo.group,name_group=distalgo.names)
>>> clf.fit(distalgo)
DISTATIS(group=(6,6,6,6),name_group=("Pixels","Measures","Ratings","Pairwise"))
"""
distalgo = namedtupledocstring(__doc__,"distalgo",distalgo.keys())(*distalgo.values())

#------------------------------------------ doubs dataset ----------------------------------------------------- 
doubs = {
    "fish" : read_excel(DATASETS_DIR/"doubs.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "env" : read_excel(DATASETS_DIR/"doubs.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"doubs.xlsx",sheet_name="Feuil3",index_col=0,header=0),
    "group" : (27,11),
    "name" : ("Fish Species","Environmental")
}
__doc__ = """
Pair of Ecological Tables

This data set gives environmental variables, fish species for 30 sites structured:

Returns
-------   
doubs : Dataset
    An object with the following attributes:
    
    fish : DataFrame of shape (30,27)
        Dataset with :math:`30` sites and :math:`27` fish species:
        
        * Cottus gobio (Cogo), 
        * Salmo trutta fario (Satr), 
        * Phoxinus phoxinus (Phph), 
        * Nemacheilus barbatulus (Neba), 
        * Thymallus thymallus (Thth), 
        * Telestes soufia agassizi (Teso), 
        * Chondrostoma nasus (Chna), 
        * Chondostroma toxostoma (Chto), 
        * Leuciscus leuciscus (Lele), 
        * Leuciscus cephalus cephalus (Lece), 
        * Barbus barbus (Baba), 
        * Spirlinus bipunctatus (Spbi), 
        * Gobio gobio (Gogo), 
        * Esox lucius (Eslu), 
        * Perca fluviatilis (Pefl), 
        * Rhodeus amarus (Rham), 
        * Lepomis gibbosus (Legi), 
        * Scardinius erythrophtalmus (Scer), 
        * Cyprinus carpio (Cyca), 
        * Tinca tinca (Titi), 
        * Abramis brama (Abbr), 
        * Ictalurus melas (Icme), 
        * Acerina cernua (Acce), 
        * Rutilus rutilus (Ruru), 
        * Blicca bjoerkna (Blbj), 
        * Alburnus alburnus (Alal), 
        * Anguilla anguilla (Anan)
    env : DataFrame of shape (30,11)
        Dataset with :math:`30` species and :math:`11` environmental variables:
        
        * dfs - distance from the source (km * 10), 
        * alt - altitude (m), 
        * slo (ln(x+1) where x is the slope (per mil * 100), 
        * flo - minimum average stream flow (m3/s * 100), 
        * pH (* 10), 
        * har - total hardness of water (mg/l of Calcium), 
        * pho - phosphates (mg/l * 100), 
        * nit - nitrates (mg/l * 100), 
        * amm - ammonia nitrogen (mg/l * 100), 
        * oxy - dissolved oxygen (mg/l * 10), 
        * bdo - biological demand for oxygen (mg/l * 10).
    data : DataFrame of shape (30,38)
        Overall dataset.
    group : tuple, default = (27,11)
        The number of columns in each group. 
    name : tuple, default = ("Fish Species","Environmental")
        The name of the columns groups.

References
----------
[1] Verneaux, J. (1973) Cours d'eau de Franche-Comté (Massif du Jura). Recherches écologiques sur le réseau hydrographique du Doubs. Essai de biotypologie. Thèse d'état, Besançon. 1–257.

Examples
--------
>>> from scientisttools.datasets import doubs
>>> from scientisttools import COIA
>>> clf = COIA(group=(27,11),type_group=("f","s"),name_group=doubs.name)
>>> clf.fit(doubs.data)
COIA(group=(27,11),type_group=("f","s"),name_group=("Fish Species","Environmental"))
"""
doubs = namedtupledocstring(__doc__,"doubs",doubs.keys())(*doubs.values())

#------------------------------------------ dune dataset ----------------------------------------------------- 
dune = read_excel(DATASETS_DIR/"dune.xlsx",sheet_name="Feuil1",index_col=0,header=0)
dune.__doc__ = """
Dune Meadow Vegetation Data

Examples
--------
>>> from scientisttools.datasets import dune
>>> from scientisttools import CCA
>>> clf = CCA(ncp=2,env=range(5),scaling=1)
>>> clf.fit(dune)
CCA(env=range(5),ncp=2,scaling=1)
"""

#------------------------------------------ femmetravail dataset ----------------------------------------------------- 
femmetravail = read_csv(DATASETS_DIR/"femmetravail.csv",delimiter=";",encoding = "cp1252",index_col =0)
femmetravail.__doc__ = """
Femmes travail Dataset

A data frame with 3 rows and 7 columns

Examples
--------
>>> from scientisttools.datasets import femmetravail
>>> from scientisttools import CA
>>> ca = CA(col_sup=range(3,7))
>>> ca.fit(femmetravail)
CA(col_sup=range(3,7))
"""

#------------------------------------------ fitnessclub dataset ----------------------------------------------------- 
fitnessclub = {
    "data" : read_excel(DATASETS_DIR/"fitnessclub.xlsx",sheet_name="Feuil1",header=0,index_col=None),
    "group" : (3,3),
    "name" : ("Physiological Measurements","Exercises"),
    "prefix" : ("Physiological","Exercises")
}
__doc__ = """
Fitness Club Dataset

Three physiological and three exercise variables are measured on 20 middle-aged men in a fitness club. 

Returns
-------   
fitnessclub : Dataset
    An object with the following attributes:
    
    data : DataFrame of shape (20,6)
        Overall dataset.
    group : tuple, default = (3,3)
        The number of columns in each group. 
    name : tuple, default = ("Physiological Measurements","Exercises")
        The name of the columns groups.
    prefix : tuple, default = ("Physiological","Exercises")
        The prefix name of the columns groups.

Examples
--------
>>> from scientisttools.datasets import fitnessclub
>>> from scientisttools import CANCORR
>>> clf = CANCORR(scale_unit=True,ncp=3,group=fitnessclub.group,name_group=fitnessclub.group,prefix_name=fitnessclub.prefix)
>>> clf.fit(fitnessclub.data)
CANCORR(group=(3,3),name_group=("Physiological Measurements","Exercises"),ncp=3,prefix_name=("Physiological","Exercises"),scale_unit=True)
"""

#------------------------------------------ friday87 dataset -----------------------------------------------------
friday87 = {
    "fau" : read_excel(DATASETS_DIR/"friday87.xlsx",sheet_name="Feuil1",index_col=0),
    "mil" : read_excel(DATASETS_DIR/"friday87.xlsx",sheet_name="Feuil2",index_col=0),
    "data" : read_excel(DATASETS_DIR/"friday87.xlsx",sheet_name="Feuil3",index_col=0),
    "group" : (11,7,13,4,13,22,4,3,8,6),
    "name" : ("Hemiptera","Odonata","Trichoptera","Ephemeroptera","Coleoptera","Diptera","Hydracarina","Malacostraca","Mollusca","Oligochaeta")
}
__doc__ = """
Faunistic K-tables dataset

The data set gives informations about sites, species and environmental variables.

Returns
-------   
friday87 : Dataset
    An object with the following attributes:
    
    fau : DataFrame of shape (16,91)
        Faunistic table with :math:`16` sites and :math:`91` species grouped as follows:
        
        * Hemiptera: 11 columns
        * Odonata: 7 columns
        * Trichoptera: 13 columns
        * Ephemeroptera: 4 columns
        * Coleoptera: 13 columns
        * Diptera: 22 columns
        * Hydracarina: 4 columns
        * Malacostraca: 3 columns
        * Mollusca: 8 columns
        * Oligochaeta: 6 columns
    mil : DataFrame of shape (16,11)
        Environmental variables.
    data : DataFrame of shape (16,102)
        Overall dataset.
    group : tuple, default = (11,7,13,4,13,22,4,3,8,6)
        The number of columns in each group (number of species per group). 
    name : tuple, default = ("Hemiptera","Odonata","Trichoptera","Ephemeroptera","Coleoptera","Diptera","Hydracarina","Malacostraca","Mollusca","Oligochaeta")
        The name of the columns groups (each group of species).
References
----------
[1] Friday, L.E. (1987) The diversity of macroinvertebrate and macrophyte communities in ponds, Freshwater Biology, 18, 87-104.

Examples
--------
>>> from scientisttools.datasets import friday87
>>> from scientisttools import MFA, CCA
>>> # multiple factor analysis (MFA)
>>> clf = MFA(group=friday87.group,group_type=("f","f","f","f","f","f","f","f","f","f"),name_group=friday87.name)
>>> clf.fit(friday87.fau)
MFA(group=(11,7,13,4,13,22,4,3,8,6),group_type=("f","f","f","f","f","f","f","f","f","f"),name_group=("Hemiptera","Odonata","Trichoptera","Ephemeroptera","Coleoptera","Diptera","Hydracarina","Malacostraca","Mollusca","Oligochaeta"))
>>> # canonical correspondence analysis (CCA)
>>> clf = CCA(env=range(91,102))
>>> clf.fit(friday87.data)
CCA(env=range(91,102))
"""
friday87 = namedtupledocstring(__doc__,"friday87",friday87.keys())(*friday87.values())

#------------------------------------------ geomorphology dataset -----------------------------------------------------
geomorphology = read_r(DATASETS_DIR/"geomorphology.rda")["geomorphology"]
geomorphology.__doc__ = """
Geomorphology Dataset

A data frame with 75 rows and 11 columns. Rows represent the individuals, columns represent the different questions. 10 variables are quantitative and one variable is qualitative.

Examples
--------
>>> from scientisttools.datasets import geomorphology
>>> from scientisttools import FAMD
>>> clf = FAMD()
>>> clf.fit(geomorphology)
FAMD()
"""

#------------------------------------------ gironde dataset -----------------------------------------------------
gironde = {
    "employment": read_r(DATASETS_DIR/"gironde_employment.rda")["employment"],
    "housing": read_r(DATASETS_DIR/"gironde_housing.rda")["housing"],
    "services": read_r(DATASETS_DIR/"gironde_services.rda")["services"],
    "environment": read_r(DATASETS_DIR/"gironde_environment.rda")["environment"],
    "data" : read_r(DATASETS_DIR/"gironde.rda")["gironde"],
    "group" : (9,5,9,4),
    "name" : ("employment","housing","services","environment")
}
__doc__ = """
Gironde Dataset

A dataset with 542 individuals and 27 columns

Examples
--------
>>> from scientisttools.datasets import gironde
>>> from scientisttools import MFA
>>> data = gironde.data.iloc[:20,:]
>>> clf = MFA(group=gironde.group,group_type=("s","m","n","s"),name_group=gironde.name)
>>> clf.fit(data)
MFA(group=(9,5,9,4),group_type=("s","m","n","s"),name_group=("employment","housing","services","environment"))
"""
gironde = namedtupledocstring(__doc__,"gironde",gironde.keys())(*gironde.values())

#------------------------------------------ housetasks dataset -----------------------------------------------------
housetasks = read_r(DATASETS_DIR/"housetasks.rda")["housetasks"]
housetasks.__doc__ = """
House tasks contingency table

A data frame containing the frequency of execution of 13 house tasks in the couple. This table is also available in ade4 R package.
A pandas DataFrame with 13 observations (house tasks) on the following 4 columns : Wife, Alternating, Husband and Jointly

Examples
--------
>>> from scientisttools.datasets import housetasks
>>> from scientisttools import CA
>>> clf = CA()
>>> clf.fit(housetasks)
CA()
"""

#------------------------------------------ housevotes84 dataset -----------------------------------------------------
housevotes84 = read_excel(DATASETS_DIR/"housevotes84.xlsx",header=0,index_col=0)
housevotes84.__doc__ = """
House Votes 84

Examples
--------
>>> from scientisttools.datasets import housevotes84
>>> from scientisttools import DMFA
>>> clf = DMFA(scale_unit=True,ncp=2,group=0,ind_sup=range(400,435))
>>> clf.fit(housevotes84)
DMFA(scale_unit=True,ncp=2,group=0,ind_sup=range(400,435))
"""

#------------------------------------------ ichtyo dataset -----------------------------------------------------
ichtyo = read_excel(DATASETS_DIR/"ichtyo.xlsx",index_col=0,header=0)
ichtyo.__doc__ = """
Point sampling of fish community

This data set gives informations between a faunistic array, the total number of sampling points made at each sampling occasion and the year of the sampling occasion.

    * faunistic: from 1 to 9 columns
    * eff : 10th column which is the sampling effort.
    * dat : 11th column where the levels are the 10 years of the sampling occasion.

Examples
--------
>>> from scientisttools.datasets import ichtyo
>>> from scientisttools import CA
>>> clf = CA(ref=9,sup_var=10)
>>> clf.fit(ichtyo)
CA(ref=9,sup_var=10)
"""

#------------------------------------------ insects dataset -----------------------------------------------------
insects = read_excel(DATASETS_DIR/"insects.xlsx",index_col=0,header=0)
insects.__doc__ = """
Insects Dataset

The data correspond to the counts of 10 species of insects on 12 different sites in a tropical region. 
A second table (displayed in red color) includes 3 quantitative variables that describe the 12 sites (altitude, humidity, and distance to the lake).

Examples
--------
>>> from scientisttools.datasets import insects
>>> from scientisttools import CCA
>>> clf = CCA(env=(10,11,12))
>>> clf.fit(insects)
CCA(env=(10,11,12))
"""

#------------------------------------------ iris dataset -----------------------------------------------------
iris = read_csv(DATASETS_DIR/"iris.csv",delimiter=",",header=0)
iris.__doc__ = """
Iris Dataset

Examples
--------
>>> from scientisttools.datasets import iris
>>> from scientisttools import DMFA
>>> clf = DMFA(group=4)
DMFA(group=4)
"""

#------------------------------------------ jobrate dataset -----------------------------------------------------
jobrate = read_excel(DATASETS_DIR/"jobrate.xlsx",index_col=None,header=0)
jobrate.__doc__ = """
Jobrate Dataset

Examples
--------
>>> from scientisttools.datasets import jobrate
>>> from scientisttools import PCA
>>> clf = PCA(sup_var=13)
>>> clf.fit(jobrate)
PCA(sup_var=13)
"""

#------------------------------------------ jobs dataset -----------------------------------------------------
jobs = {
    "data" : read_excel(DATASETS_DIR/"jobs.xlsx",index_col=None,header=0),
    "group" : (3,3),
    "name" : ("Satisfaction Areas","Job Characteristics"),
    "prefix" : ("Satisfaction","Characteristics")
}
__doc__ = """
Jobs Dataset

Your three variables associated with job satisfaction are as follows:

    * career track satisfaction: employee satisfaction with career direction and the possibility of future advancement, expressed as a percent
    * management and supervisor satisfaction: employee satisfaction with supervisor's communication and management style, expressed as a percent
    * financial satisfaction: employee satisfaction with salary and other benefits, using a scale measurement from 1 to 10 (1=unsatisfied, 10=satisfied)

The three variables associated with job characteristics are as follows:

    * task variety: degree of variety involved in tasks, expressed as a percent
    * feedback: degree of feedback required in job tasks, expressed as a percent
    * autonomy: degree of autonomy required in job tasks, expressed as a percent
    
Returns
-------   
jobs : Dataset
    An object with the following attributes:
    
    data : DataFrame of shape (20,24)
        Overall dataset.
    group : tuple, default = (3,3)
        The number of columns in each group. 
    name : tuple, default = ("Satisfaction Areas","Job Characteristics")
        The name of the columns groups.
    prefix : tuple, default = ("Satisfaction","Characteristics")
        The prefix name of the columns groups.

Examples
--------
>>> from scientisttools.datasets import jobs
>>> from scientisttools import CANCORR
>>> clf = CANCORR(scale_unit=False,ncp=3,group=jobs.group,name_group=jobs.name,prefix_group=jobs.prefix)
>>> clf.fit(jobs.data)
CANCORR(group=(3,3),name_group=("Satisfaction Areas","Job Characteristics"),ncp=3,prefix_group=("Satisfaction","Characteristics"),scale_unit=False)
"""
jobs = namedtupledocstring(__doc__,"jobs",jobs.keys())(*jobs.values())

#------------------------------------------ lifecyclesavings dataset -----------------------------------------------------
lifecyclesavings = read_r(DATASETS_DIR/"LifeCycleSavings.RData")["LifeCycleSavings"]
lifecyclesavings.__doc__ = """
Intercountry Life-Cycle Savings Data

Data on the savings ratio 1960 - 1970
A data frame with 50 observations on 5 variables

Source
------
The LifeCycle Savings dataset from R datasets

Examples
--------
>>> from scientisttools.datasets import lifecyclesavings
>>> from scientisttools import CCA
>>> clf = CCA(env=(1,2))
>>> clf.fit(lifecyclesavings)
CCA(env=(1,2))
"""

#------------------------------------------ loisirs dataset -----------------------------------------------------
loisirs = read_excel(DATASETS_DIR/"loisirs_subset.xlsx",sheet_name="data",header=0)
loisirs.__doc__ = """
Loisirs Dataset

Examples
--------
>>> from scientisttools.datasets import loisirs
>>> from scientisttools import MCA, CatVARHCPC
>>> clf = MCA(ncp=2)
>>> clf.fit(loisirs)
>>> clf2 = CatVARHCPC(ncl=3,method="average")
>>> clf2.fit(clf)
"""

#------------------------------------------ madagascar dataset -----------------------------------------------------
madagascar = read_excel(DATASETS_DIR/"madagascar.xlsx",index_col=0,header=0)
madagascar.__doc__ = """
Madagascar Dataset

Examples
--------
>>> from scientisttools.datasets import madagascar
>>> from scientisttools import PCoA
>>> clf = PCoA(metric="precomputed")
>>> clf.fit(madagascar)
PCoA(metric ="precomputed")
"""

#------------------------------------------ meaudret dataset ----------------------------------------------------- 
meaudret = {
    "actif" : read_excel(DATASETS_DIR/"meaudret.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"meaudret.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"meaudret.xlsx",sheet_name="Feuil3",index_col=0,header=0)
}
__doc__ = """
Meaudret Dataset

Ecological Data: sites-variables, sites-species, where and when
This data set contains information about sites, environmental variables and Ephemeroptera Species.

Returns
-------   
meaudret : Dataset
    An object with the following attributes:
    
    actif : DataFrame of shape (20,10)
        Actif dataset.
    sup_var : DataFrame of shape (20,14)
        Supplementary variables.
    data : DataFrame of shape (20,24)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import meaudret
>>> from scientisttools import PCA
>>> clf = PCA(group=9,option="within")
>>> clf.fit(meaudret.actif)
PCA(group=9,option="within"")
"""
meaudret = namedtupledocstring(__doc__,"meaudret",meaudret.keys())(*meaudret.values())

#------------------------------------------ mortality dataset -----------------------------------------------------  
mortality = {
    "data" : read_r(DATASETS_DIR/"mortality.rda")["mortality"],
    "group" : (9,9),
    "name" : ("y1979","y2006")
}
__doc__ = """
Mortality Dataset

The cause of mortality in France in 1979 and 2006.

A data frame with 62 rows (the different causes of death) and 18 columns. Each column corresponds to an age interval (15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85-94, 95 and more) in a year. 
The 9 first columns correspond to data in 1979 and the 9 last columns to data in 2006. In each cell, the counts of deaths for a cause of death in an age interval (in a year) is given.

Returns
-------
mortality : Dataset
    An object with the following attributes:
    
    data : DataFrame of shape (62,18)
        Overall dataset.
    group : tuple, default = (9,9)
        The number of columns in each group.
    name : tuple, default = ("y1979","y2006")
        The name of each group.

Examples
--------
>>> from scientisttools.datasets import mortality
>>> from scientisttools import MFA
>>> clf = MFA(group=mortality.group,group_type=("f","f"),name_group=mortality.name)
>>> clf.fit(mortality.data)
MFA(group=(9,9),group_type=("f","f"),name_group=("y1979","y2006"))
"""
mortality = namedtupledocstring(__doc__,"mortality",mortality.keys())(*mortality.values())

#------------------------------------------ mushroom dataset -----------------------------------------------------
mushroom = read_excel(DATASETS_DIR/"mushroom.xlsx")
mushroom.__doc__ = """
Mushroom Dataset

Examples
--------
>>> from scientisttools.datastes import mushroom
>>> from scientisttools import MCA
>>> clf = MCA()
>>> clf.fit(mushroom)
MCA()
"""

#------------------------------------------ music dataset ----------------------------------------------------- 
music = read_r(DATASETS_DIR/"music.RData")["Music"]
music.__doc__ = """
Music Dataset

The data concerns tastes for music of a set of 500 individuals. It contains :
    * 5 variables of likes for music genres (french pop, rap, rock, jazz and classical), 
    * 2 variables about music listening and,
    * 2 additional variables (gender and age).

Examples
--------
>>> from scientisttools.datasets import music
>>> from scientisttools import MCA
>>> clf = MCA(excl=(0,3,6,9,12),sup_var=(5,6,7,8))
>>> clf.fit(music)
MCA(excl=(0,3,6,9,12),sup_var=(5,6,7,8))
"""

#------------------------------------------ oliveoil dataset ----------------------------------------------------- 
oliveoil = read_excel(DATASETS_DIR/"oliveoil.xlsx",index_col=0,header=0)
oliveoil.__doc__ = """
Sensory and physico-chemical data of olive oils

A data set with scores on:
    * 6 attributes from a sensory panel : yellow, green, brown, glossy, transp, and syrup.
    * measurements of 5 physico-chemical quality parameters: acidity, peroxide, K232, K270, and DK.

on 16 olive oil samples:
    * The first five oils are Greek, 
    * the next five are Italian and,
    * the last six are Spanish.

Examples
--------
>>> from scientisttools.datasets import oliveoil
>>> from scientisttools import PCA
>>> clf = PCA()
>>> clf.fit(oliveoil)
PCA()
"""

#------------------------------------------ olympic dataset -----------------------------------------------------
olympic = {
    "actif": read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "ind_sup": read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "sup_var": read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil3",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil4",index_col=0,header=0)
}
__doc__ = """
Olympic Decathlon Dataset

This data set gives the performances of 33 men's decathlon at the Olympic Games (1988) and 5 men's decathlon at the Olympic Games (2004)

A data frame with 38 rows and 12 columns: the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. 
The columns 11 and 12 correspond respectively to the rank and the points obtained.
Supplementary individuals are the top 5 from the 2004 decathlon Olympic Games.

Returns
-------   
olympic : Dataset
    An object with the following attributes:
    
    actif : DataFrame of shape (33,10)
        Actif dataset.
    ind_sup : DataFrame of shape (5,10)
        Supplementary individuals.
    sup_var : DataFrame of shape (33,2)
        Supplementary variables.
    data : DataFrame of shape (38,12)
        Overall dataset.
    
Examples
--------
>>> from scientisttools.datasets import olympic
>>> from scientisttools import PCA
>>> clf = PCA(ind_sup=(33,34,35,36,37),sup_var=(10,11))
>>> clf.fit(olympic.data)
PCA(ind_sup=(33,34,35,36,37),sup_var=(10,11))
"""
olympic = namedtupledocstring(__doc__,"olympic",olympic.keys())(*olympic.values())

#------------------------------------------ poison dataset -----------------------------------------------------
poison = {
    "actif" : read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil1",index_col=0,header=0),
    "sup_var" : read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil2",index_col=0,header=0),
    "data" : read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil3",index_col=0,header=0),
    "group" : (2,2,5,6),
    "name" : ("desc","desc2","symptom","eat")
}
__doc__ = """
Poison Dataset

The data used here refer to a survey carried out on a sample of children of primary school who suffered from food poisoning. 
They were asked about their symptoms and about what they ate.
A data frame with 55 rows and 15 columns.

Returns
-------   
poison : Dataset
    An object with the following attributes:
    
    actif : DataFrame of shape (55,11)
        Actif dataset
    sup_var : DataFrame of shape (55,4)
        Supplementary variables.
    data : DataFrame of shape (55,15)
        Overall dataset.
    group : tuple, default = (2,2,5,6)
        The number of columns in each group. 
    name_group : tuple, default = ("desc","desc2","symptom","eat")
        The name of the columns groups.

References
----------
[1] Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. Journal of Statistical Software, 25(1), 1-18. https://doi.org/10.18637/jss.v025.i01

Examples
--------
>>> from scientisttools.datasets import poison
>>> from scientisttools import MCA
>>> # multiple correspondence analysis (MCA)
>>> clf = MCA(sup_var=range(4))
>>> clf.fit(poison.data)
MCA(sup_var=range(4))
>>> # multiple factor analysis (MFA)
>>> clf = MFA(group=poison.group,group_type=("s","n","n","n"),name_group=poison.name,num_group_sup=(0,1))
>>> clf.fit(poison.data)
MFA(group=(2,2,5,6),group_type=("s","n","n","n"),name_group=("desc","desc2","symptom","eat"),num_group_sup=(0,1))
"""
poison = namedtupledocstring(__doc__,"poison",poison.keys())(*poison.values())

#------------------------------------------ protein dataset -----------------------------------------------------
protein = read_r(DATASETS_DIR/"protein.RData")["protein"]
protein.__doc__ = """
Protein dataset

This dataset gives the amount of protein consumed for nine food groups in 25 European countries. The nine food groups are:
    1. red meat (RedMeat), 
    2. white meat (WhiteMeat), 
    3. eggs (Eggs), 
    4. milk (Milk), 
    5. fish (Fish), 
    6. cereal (Cereal), 
    7. starch (Starch), 
    8. nuts (Nuts), and 
    9. fruits and vegetables (FruitVeg).
A numerical data matrix with 25 rows (the European countries) and 9 columns (the food groups)

References
----------
[1] The protein dataset for sparsePCA R package

Examples
--------
>>> from scientisttools.datasets import protein
>>> from scientisttools import PCA
>>> clf = PCA()
>>> clf.fit(protein)
PCA()
"""

#------------------------------------------ qtevie dataset -----------------------------------------------------
qtevie = {
    "data" : read_csv(DATASETS_DIR/"qtevie.csv",encoding="ISO-8859-1",header=0,sep=";",index_col=0),
    "group" : (5,5,3,6,3,1),
    "name" : ("Bien-être matériel","Emploi","Satisfaction","Santé et sécurité","Enseignement","Région")
}
__doc__ = """
Qualité de vie Dataset

34 country of OCDE with Russia and Brazil describe by 22 indicators and one qualitative variable group by theme : 
    * material.well.being (5), 
    * employment (5), 
    * satisfaction (3), 
    * health.and.safety (6), 
    * education (3)
    * region (1)
    
Returns
-------
qtevie : Dataset
    An object with the following attributes:
    
    data : DataFrame of shape ()
        Overall dataset.
    group : tuple, default = (5,5,3,6,3,1)
        The number of columns in each group.
    name : tuple, default = ("Bien-être matériel","Emploi","Satisfaction","Santé et sécurité","Enseignement","Région")
        The name of each group.

Source
------
OCDE

Examples
--------
>>> from scientisttools.datasets import qtevie
>>> from scientisttools import MFA
>>> clf = MFA(group=qtevie.group,name_group=qtevie.name,group_type=("s","s","s","s","s","n"),num_group_sup=5)
>>> clf.fit(qtevie.data)
MFA(group=(5,5,3,6,3,1),name_group=("Bien-être matériel","Emploi","Satisfaction","Santé et sécurité","Enseignement","Région"),group_type=("s","s","s","s","s","n"),num_group_sup=5)
""" 
qtevie = namedtupledocstring(__doc__,"qtevie",qtevie.keys())(*qtevie.values())

#------------------------------------------ rhone dataset -----------------------------------------------------
rhone = {
    "actif" : read_excel(DATASETS_DIR/"rhone.xlsx",sheet_name="Feuil1",header=0),
    "sup_var" : read_excel(DATASETS_DIR/"rhone.xlsx",sheet_name="Feuil2",header=0),
    "data" : read_excel(DATASETS_DIR/"rhone.xlsx",sheet_name="Feuil3",header=0)
}
__doc__ = """
Physico-Chemistry Dataset

This data set gives for 39 water samples a physico-chemical description with the number of sample date and the flows of three tributaries.

Returns
-------
rhone : Dataset
    An object with the following attributes:
    
    actif: DataFrame of shape (39,15)
        Active dataset.
    sup_var: DataFrame of shape (39,3)
        Supplementary variables.
    data: DataFrame of shape (39,18)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import rhone
>>> from scientisttools import PCA
>>> clf = PCA(var_sup=(15,16,17))
>>> clf.fit(rhone.data)
PCA(var_sup=(15,16,17))
"""
rhone = namedtupledocstring(__doc__,"rhone",rhone.keys())(*rhone.values())

#------------------------------------------ rpjdl dataset -----------------------------------------------------
rpjdl = read_excel(DATASETS_DIR/"rpjdl.xlsx",header=0,index_col=0)
rpjdl.__doc__ = """
Avifauna and Vegetation

This data set gives the abundance of 51 species and 8 environmental variables in 182 sites.

Examples
---------
>>> from scientisttools.datasets import rpjdl
>>> from scientisttools import CA
>>> clf = CA(iv=range(51,58))
>>> clf.fit(rpjdl)
CA(iv=range(51,58))
"""

#------------------------------------------ sales dataset -----------------------------------------------------
sales = {
    "data" : read_csv(DATASETS_DIR/"sales.csv"),
    "group" : (3,4),
    "name" : ("Sales Performance","Test Scores"),
    "prefix" : ("sales","scores")
}
__doc__ = """
Sales Data

The example data comes from a firm that surveyed a random sample of n = 50 of its employees in an attempt to determine which factors influence sales performance. 
Two collections of variables were measured:

Sales Performance:
    * Sales Growth
    * Sales Profitability
    * New Account Sales
Test Scores as a Measure of Intelligence
    * Creativity
    * Mechanical Reasoning
    * Abstract Reasoning
    * Mathematics
There are p = 3 variables in the first group relating to Sales Performance and q = 4 variables in the second group relating to Test Scores.

Returns
-------
sales : Dataset
    An object with the following attributes:
    
    data: DataFrame of shape (15,12)
        Overall dataset.
    group: tuple, default = (3,4)
        The number of columns in each group. 
    name: tuple, default = ("sales.performance","test.scores")
        The name of the columns groups.
    prefix: tuple, default = ("sales","scores")
        The prefix name of the columns groups.

Examples
--------
>>> from scientisttools.datasets import sales
>>> from scientisttools import CANCORR
>>> clf = CANCORR(scale_unit=False,ncp=3,group=sales.group,name_group=sales.name,prefix_group=sales.prefix)
>>> clf.fit(sales.data)
CANCORR(scale_unit=False,ncp=3,group=(3,4),name_group=("Sales Performance","Test Scores"),prefix_group=("sales","scores"))
"""
sales = namedtupledocstring(__doc__,"sales",sales.keys())(*sales.values())

#------------------------------------------- tea dataset -----------------------------------------------------
tea = read_r(DATASETS_DIR/"tea.rda")["tea"]
tea.__doc__ = """
Tea

The data used here concern a questionnaire on tea. We asked to 300 individuals how they drink tea (18 questions), what are their product's perception (12 questions) and some personal details (4 questions).

Returns
-------
A dataframe with 300 rows and 36 columns. Rows represent the individuals, columns represent the different questions. The first 18 questions are active ones, the 19th is a supplementary quantitative variable (the age) and the last variables are supplementary categorical variables.

References
----------
[1] The tea dataset from FactoMineR.

Examples
--------
>>> from scientisttools.datasets import tea
>>> from scientisttools import MCA
>>> clf = MCA(sup_var=range(18,36))
>>> clf.fit(tea)
MCA(sup_var=range(18,36))
"""

#------------------------------------------ temperature dataset ----------------------------------------------------- 
temperature = {
    "actif" : read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil1",header=0,index_col=0),
    "ind_sup" : read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil2",header=0,index_col=0),
    "sup_var" : read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil3",header=0,index_col=0),
    "data" : read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil4",header=0,index_col=0)
}
__doc__ = """
Temperature Dataset

Returns
-------
temperature : Dataset
    An object with the following attributes:
    
    actif: DataFrame of shape (15,12)
        Active dataset.
    ind_sup: DataFrame of shape (2,12)
        Supplementary individuals.
    sup_var: DataFrame of shape (15,5)
        Supplementary variables.
    data: DataFrame of shape (17,17)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import temperature
>>> from scientisttools import PCA
>>> clf = PCA(ind_sup=range(15,25),sup_var=range(12,17))
>>> clf.fit(temperature.data)
PCA(ind_sup=range(15,25),sup_var=range(12,17))
"""
temperature = namedtupledocstring(__doc__,"temperature",temperature.keys())(*temperature.values())

#------------------------------------------ tennis dataset ----------------------------------------------------- 
tennis = {
    "actif" : read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil1",header=0,index_col=0),
    "ind_sup" : read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil2",header=0,index_col=0),
    "sup_var" : read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil3",header=0,index_col=0),
    "data" : read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil4",header=0,index_col=0)
}
__doc__ = """
Tennis 2020 Dataset

Returns
-------
tennis : Dataset
    An object with the following attributes:
    
    actif: DataFrame of shape (16,7)
        Active dataset.
    ind_sup: DataFrame of shape (4,7)
        Supplementary individuals.
    sup_var: DataFrame of shape (16,1)
        Supplementary variables.
    data: DataFrame of shape (20,8)
        Overall dataset.

Examples
--------
>>> from scientisttools.datasets import tennis
>>> from scientisttools import FAMD
>>> clf = FAMD(ncp=2,ind_sup=range(16,20),sup_var=7)
>>> clf.fit(tennis.data)
FAMD(ind_sup=range(16,20),ncp=2,sup_var=7)
"""
tennis = namedtupledocstring(__doc__,"tennis",tennis.keys())(*tennis.values())

#------------------------------------------ universite dataset -----------------------------------------------------
universite = read_csv(DATASETS_DIR/"universite.csv",delimiter=";",header=0)
universite.__doc__ = """
Universite Dataset

Examples
--------
>>> from scientisttools.datasets import universite
>>> from scientisttools import CA
>>> clf = CA(col_sup=range(6,12))
>>> clf.fit(universite)
CA(col_sup=range(6,12))
"""

#------------------------------------------ usarrests dataset ----------------------------------------------------- 
usarrests = read_excel(DATASETS_DIR/"usarrests.xlsx",index_col=0,header=0)
usarrests.__doc__ = """
Violent Crime Rates by US State

This data set contains statistics, in arrests per 100,000 residents for assault, murder, and rape in each of the 50 US states in 1973. 
Also given is the percent of the population living in urban areas.
dataframe with 50 observations on 4 variables:

    * Murder : numeric Murder arrests (per 100,000)
    * Assault : numeric Assault arrests (per 100,000)
    * UrbanPop : numeric Percent urban population
    * Rape : numeric Rape arrests (per 100,000)

Source
------
World Almanac and Book of facts 1975. (Crime rates).

Statistical Abstracts of the United States 1975, p.20, (Urban rates), possibly available as https://books.google.ch/books?id=zl9qAAAAMAAJ&pg=PA20.

References
----------
McNeil, D. R. (1977) Interactive Data Analysis. New York: Wiley.

Examples
--------
>>> from scientisttools.datasets import usarrests
>>> from scientisttools import PCA
>>> clf = PCA()
>>> clf.fit(usarrests)
PCA()
"""

#------------------------------------------ uscrime dataset ----------------------------------------------------- 
uscrime = read_excel(DATASETS_DIR/"uscrime.xlsx",sheet_name="Feuil1",header=0,index_col=0)
uscrime.__doc__ = """
US Crime Dataset

These data are crime-related and demographic statistics for 47 US states in 1960. 
The data were collected from the FBI's Uniform Crime Report and other government agencies to determine how the variable crime rate depends on the other variables measured in the study.

    1. Crime.rate: # of offenses reported to police per million population
    2. Male14_24: The number of males of age 14-24 per 1000 population
    3. Southern.states: Indicator variable for Southern states (Yes, No)
    4. Education: Mean # of years of schooling x 10 for persons of age 25 or older
    5. Expend60: 1960 per capita expenditure on police by state and local government
    6. Expend59: 1959 per capita expenditure on police by state and local government
    7. Labor.force: Labor force participation rate per 1000 civilian urban males age 14-24
    8. Male: The number of males per 1000 females
    9. Pop.size: State population size in hundred thousands
    10. Non.white: The number of non-whites per 1000 population
    11. Unemp14_24: Unemployment rate of urban males per 1000 of age 14-24
    12. Unemp35_39: Unemployment rate of urban males per 1000 of age 35-39
    13. Family.income: Median value of transferable goods and assets or family income in tens of $
    14. Under.median: The number of families per 1000 earning below 1/2 the median income

References
----------
[1] see https://lib.stat.cmu.edu/DASL/Datafiles/USCrime.html

[2] Vandaele, W. (1978) Participation in illegitimate activities: Erlich revisited. In Deterrence and incapacitation, Blumstein, A., Cohen, J. and Nagin, D., eds., Washington, D.C.: National Academy of Sciences, 270-335. Methods: A Primer, New York: Chapman & Hall, 11. Also found in: Hand, D.J., et al. (1994) A Handbook of Small Data Sets, London: Chapman & Hall, 101-103.

Examples
--------
>>> from scientisttools.datasets import uscrime
>>> from scientisttools import FAMD
>>> clf = FAMD()
>>> clf.fit(uscrime)
FAMD()
"""

#------------------------------------------ vegetation dataset ----------------------------------------------------- 
vegetation = read_excel(DATASETS_DIR/"vegetation.xlsx",header=0,index_col=0)
vegetation.__doc__ = """
Vegetation Dataset

Examples
--------
>>> from scientisttools.datasets import vegetation
>>> from scientisttools import CCA
>>> clf = CCA(env=range(44,58))
>>> clf.fit(vegetation)
CCA(env=range(44,58))
"""

#------------------------------------------ villes dataset -----------------------------------------------------
villes = read_csv(DATASETS_DIR/"villes.csv",delimiter=";",header=0)
villes.__doc__ = """
Villes Dataset

Examples
--------
>>> from scientisttools.datasets import villes
>>> from scientisttools import PCA
>>> clf = PCA()
>>> clf.fit(villes)
PCA()
"""

#------------------------------------------ vote dataset -----------------------------------------------------
vote = read_csv(DATASETS_DIR/"vote.csv",delimiter=",",header=0)
vote.__doc__ = """
Vote Dataset

Examples
--------
>>> from scientisttools.datasets import vote
>>> from scientisttools import MCA
>>> clf = MCA(sup_var=0)
>>> clf.fit(vote)
MCA(sup_var=0)
"""

#------------------------------------------ votingrecords dataset ----------------------------------------------------- 
votingrecords = read_excel(DATASETS_DIR/"votingrecords.xlsx")
votingrecords.__doc__ = """
Congressional Voting Records Dataset

The `Congressional Voting Records <https://archive.ics.uci.edu/dataset/105/congressional+voting+records>`_ UCI dataset.

Examples
--------
>>> from scientisttools.datasets import votingrecords
>>> from scientisttools import MCA
>>> clf = MCA(sup_var=0)
>>> clf.fit(votingrecords)
MCA(sup_var=0)
"""

#------------------------------------------ wine dataset ----------------------------------------------------- 
wine = {
    "data" : read_r(DATASETS_DIR/"wine.rda")["wine"],
    "group" : (2,5,3,10,9,2),
    "name" : ("origin","odor","visual","odor.after.shaking","taste","overall")
}
__doc__ = """
Wine Dataset

The data used here refer to :math:`21` wines of Val de Loire and :math:`31` columns:

    * The first column corresponds to the label of origin.
    * The second column corresponds to the soil.
    * and the others correspond to sensory descriptors.
    
Returns
-------   
wine : Dataset
    An object with the following attributes:

    data : DataFrame of shape (21,31)
        Wine dataset.
    group : tuple, default = (2,5,3,10,9,2)
        The number of columns in each group. 
    name : tuple, default = ("origin","odor","visual","odor.after.shaking","taste","overall")
        The name of the columns groups.

References
----------
[1] Lê, S., Josse, J., & Husson, F. (2008). FactoMineR: An R Package for Multivariate Analysis. Journal of Statistical Software, 25(1), 1-18. https://doi.org/10.18637/jss.v025.i01

Examples
--------
>>> from scientisttools.datasets import wine
>>> from scientisttools import MFA
>>> clf = MFA(group=wine.group,group_type=("n","s","s","s","s","s"),name_group=wine.name,num_group_sup=(0,5))
>>> clf.fit(wine.data)
MFA(group=(2,5,3,10,9,2),group_type=("n","s","s","s","s","s"),name_group=("origin","odor","visual","odor.after.shaking","taste","overall"),num_group_sup=(0,5))
"""
wine = namedtupledocstring(__doc__,"wine",wine.keys())(*wine.values())

#------------------------------------------ winequality dataset -----------------------------------------------------
winequality = read_excel(DATASETS_DIR/"winequality.xlsx",header=0,index_col=0)
winequality.__doc__ = """
Vine Quality Dataset

Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. 

Examples
--------
>>> from scientisttools.datasets import winequality
>>> from scientisttools import mgPCA
>>> clf = mgPCA(scale_unit=True,group=0)
>>> clf.fit(winequality.drop(columns=["quality"]))
mgPCA(scale_unit=True,group=0)
"""

#------------------------------------------ womenwork dataset ----------------------------------------------------- 
womenwork = read_csv(DATASETS_DIR/"womenwork.txt",sep="\t")
womenwork.__doc__ = """
Women Work

A data with 3 rows and 7 columns.

Examples
--------
>>> from scientisttools.datasets import womenwork
>>> from scientisttools import CA
>>> clf = CA(col_sup=(3,4,5,6))
>>> clf.fit(womenwork)
CA(col_sup=(3,4,5,6))
"""

# store
__all__ = [
    "ardeche",
    "autos1990",
    "autos2005",
    "autos2006",
    "autosmds",
    "autosmds2",
    "beer",
    "body",
    "burger",
    "burgundywines",
    "canines",
    "children",
    "cultural",
    "decathlon",
    "decathlon2",
    "distalgo",
    "doubs",
    "dune",
    "femmetravail",
    "fitnessclub",
    "friday87",
    "geomorphology",
    "gironde",
    "housetasks",
    "housevotes84",
    "ichtyo",
    "insects",
    "iris",
    "jobrate",
    "jobs",
    "lifecyclesavings",
    "loisirs",
    "madagascar",
    "meaudret",
    "mortality",
    "mushroom",
    "music",
    "oliveoil",
    "olympic",
    "poison",
    "protein",
    "qtevie",
    "rhone",
    "rpjdl",
    "sales",
    "tea",
    "temperature",
    "tennis",
    "universite",
    "usarrests",
    "uscrime",
    "vegetation",
    "villes",
    "vote",
    "votingrecords",
    "wine",
    "winequality",
    "womenwork"
]