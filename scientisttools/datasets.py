# -*- coding: utf-8 -*-
from __future__ import annotations

from pandas import DataFrame, read_excel, read_csv
from pyreadr import read_r
import pathlib

# https://husson.github.io/data.html
# https://r-stat-sc-donnees.github.io/liste_don.html

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"

def load_autos1990():
    """
    Autos 1990 Dataset
    ------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_carsacpm
    >>> cars = load_carsacpm()
    ```

    Format
    ------
    a data frame with 27 individuals and 9 variables

    Reference
    ---------
    Abdesselam, R. (2006), Analyise en Composantes Principales Mixte, 

    Examples
    --------
    ```python
    >>> #load autos1990 dataset
    >>> from scientisttools import load_autos1990
    >>> autos1990 = load_autos1990() 
    >>> from scientisttools import MPCA
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(autos1990)
    ```
    """
    return read_csv(DATASETS_DIR/'autos1990.txt', delimiter = " ",header=0,index_col=0)

def load_autos2005(element="all"):
    """
    Autos 2005 Dataset
    ------------------

    Description
    -----------
    The dataset contains 45 autos

    Usage
    -----
    ```python
    >>> from scientisttools import load_autos2005
    >>> autos2005 = laod_autos2005()
    ```

    Examples
    --------
    ```python
    >>> #load autos2005 dataset
    >>> from scientisttools import load_autos2005
    >>> autos2005 = load_autos2005()
    >>> from scientisttools import FAMD
    >>> res_famd = FAMD(ind_sup=(38,39,40,41,42,43,44),quanti_sup=(12,13,14),quali_sup=15)
    >>> res_famd.fit(autos2005)
    ```
    """
    if element == "actif":
        return read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil1",index_col=0,header=0)
    elif element == "ind_sup":
        return read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil2",index_col=0,header=0)
    elif element == "quali_sup":
        return read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil3",index_col=0,header=0)
    elif element == "quali_sup":
        return read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil4",index_col=0,header=0)
    elif element == "all":
        return read_excel(DATASETS_DIR/"autos2005.xlsx",sheet_name="Feuil5",index_col=0,header=0)
    else:
        raise ValueError("'element' must be one of 'all', 'actif', 'ind_sup', 'quanti_sup', 'quali_sup'")

def load_autos2006(element="actif"):
    """
    Autos 2006 Dataset
    ------------------

    Description
    -----------
    The dataset contains 20 autos

    Usage
    -----
    ```python
    >>> from scientisttools import load_autos2006
    >>> autos2006 = load_autos2006()
    ```

    Parameters
    ----------
    `element`: the element to subset from the output. Allowed values are :
        * "actif" for actifs elements
        * "ind_sup" for supplementary individuals
        * "quanti_sup" for supplementary quantitative variables
        * "quali_sup" for supplementary qualitative variables
        * "all" for actifs and supplementary elements
    
    Returns
    -------
    pandas Dataframe

    References
    ----------
    * Saporta G. (2006). Probabilites, Analyse des données et Statistiques. Technip

    * Rakotomalala R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.0

    Examples
    --------
    ```python
    >>> #load autos2006 dataset
    >>> from scientisttools import load_autos2006
    >>> autos2006 = load_autos2006()
    >>> from scientisttools import PartialPCA, summaryPartialPCA
    >>> res_ppca = PartialPCA(partial=0,index=(18,19),quanti_sup=(6,7),quali_sup=8)
    >>> res_ppca.fit(cars)
    >>> summaryPartialPCA(res_ppca)
    ```
    """     
    if element == "actif":
        return read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil1",index_col=0,header=0)
    elif element == "ind_sup":
        return read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil2",index_col=0,header=0)
    elif element == "quanti_sup":
        return read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil3",index_col=0,header=0)
    elif element == "quali_sup":
        return read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil4",index_col=0,header=0)
    elif element == "all":
        return read_excel(DATASETS_DIR/"autos2006.xlsx",sheet_name="Feuil5",index_col=0,header=0)
    else:
        raise ValueError("'element' must be one of 'all', 'actif', 'ind_sup', 'quanti_sup', 'quali_sup'")

def load_autosmds():
    """
    Autos Multidimensional Scaling Dataset
    --------------------------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_autosmds
    >>> autosmds = load_autosmds()
    ```

    Examples
    --------
    ```python
    >>> #load autosmds dataset
    >>> from scientisttools import load_autosmds
    >>> autosmds = load_autosmds()
    >>> from scientisttools import CMDSCALE
    >>> my_cmds = CMDSCALE(n_components=2,ind_sup=(12,13,14),proximity="euclidean",normalized_stress=True,parallelize=False)
    >>> my_cmds.fit(autosmds)
    ```
    """
    return read_excel(DATASETS_DIR/"autosmds.xlsx",index_col=0,header=0)

def load_beer():
    """
    Beer Dataset
    ------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_beer
    >>> beerpfa = load_beer()
    ```

    Examples
    --------
    ```python
    >>> #load beerpfa dataset
    >>> from scientisttools import load_beer
    >>> beer = load_beer()
    >>> from scientisttools import PFA
    >>> my_pfa = PFA()
    >>> my_pfa.fit(beer)
    ```
    """
    return read_excel(DATASETS_DIR/"beer_rnd.xlsx",index_col=None,header=0)

def load_body():
    """
    Body Dimensions Datasets
    ------------------------

    Description
    -----------
    The data give some body dimension measurements as well as age, weight, height, and gender on 507 individuals. The 247 men and 260 women were primarily individuals in their twenties and thirties, with a scattering of older men and women, all exercising serveral hours a week. 

    Usage
    -----
    ```python
    >>> from scientisttools import load_body
    >>> body = load_body()
    ```

    Returns
    -------
    dataframe with 507 observations and 15 variables :

    shoulder.girth : shoulder girth (in cm) -- épaule (fr)

    chest.girth : Chest girth (in cm) -- poitrine (fr)

    waist.girth : Waist girth (in cm) -- taille (fr)

    navel.girth : Navel girth (in cm) -- nombril (fr)

    hip.girth : Hip girth (in cm) -- hanche (fr)

    thigh.girth : Thigh girth (in cm) -- cuisse (fr)

    bicep.girth : Bicep girth (in cm) -- biceps (fr)

    forearm.girth : Forearm girth (in cm) -- avant-bras (fr)

    knee.girth : Knee girth (in cm) -- genou (fr)

    calf.girth : Calf girth (in cm) -- mollet (fr)

    ankle.girth : Ankle girth (in cm) -- cheville (fr)

    wrist.girth : Wrist girth (in cm)  -- poignet (fr)

    weight : Weight (in kg)

    height : Height (in cm)

    gender : Gender ; 1 for males and 0 for females.

    Examples
    --------
    ```python
    >>> # Load dataset
    >>> from scientisttools import load_body
    >>> data = load_body()
    >>> # Drop gender
    >>> body = data.drop(columns=["gender"])
    >>> body.columns = [x.replace(".","_") for x in body.columns]
    >>> # Concatenate
    >>> import pandas as pd
    >>> D = pd.concat((body,data.drop(columns=["weight","height"])),axis=1)
    >>> from scientisttools import PartialPCA
    >>> res_partialpca = PartialPCA(standardize=True,partial=["weight","height"],quanti_sup=list(range(14,26)),quali_sup=26,parallelize=False)
    >>> res_partialpca.fit(D)
    ```
    """
    return read_excel(DATASETS_DIR/"body.xls",sheet_name="body")

def load_burgundywines():
    """
    Burgundy Wines Dataset
    ----------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_burgundywines
    >>> burgundywines = load_burgundywines()
    ```

    Source
    ------ 
    https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf

    Examples
    --------
    ```python
    >>> # Load burgundywines
    >>> from scientisttools import load_burgundywines
    >>> wines = load_burgundywines()
    >>> from scientisttools import MFA
    >>> name = ["Type","Expert1","Expert2","Expert3"] 
    >>> group_type = ["s","s","s","s"]
    >>> res_mfa = MFA(n_components=5,group=[1,3,4,3],name_group=name,group_type=group_type,num_group_sup=0)
    >>> res_mfa.fit(wines)
    ```
    """
    wines = DataFrame(
        data=[
            [1, 6, 7, 2, 5, 7, 6, 3, 6, 7],
            [5, 3, 2, 4, 4, 4, 2, 4, 4, 3],
            [6, 1, 1, 5, 2, 1, 1, 7, 1, 1],
            [7, 1, 2, 7, 2, 1, 2, 2, 2, 2],
            [2, 5, 4, 3, 5, 6, 5, 2, 6, 6],
            [3, 4, 4, 3, 5, 4, 5, 1, 7, 5],
        ],
        columns= ["Fruity.one","Woody.one","Coffee","Red fruit","Roasted","Vanillin","Woody.two","Fruity.three","Butter","Woody.three"],
        index=[f"Wine {i + 1}" for i in range(6)],
    )
    wines.insert(0, "Oak type", [1, 2, 2, 2, 1, 1])
    return wines

def load_children2(element="all"):
    """
    Children2 dataset
    -----------------

    Description
    -----------
    The data used here is a contingency table that summarizes the answers given by different categories of people to the following question : according to you, what are the reasons that can make hesitate a woman or a couple to have children?

    Usage
    -----
    ```python
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2 = load_children2()
    ```

    Parameters
    ----------
    `element`: the element to return (default "all"). Allowed values are :
        * "all" for actives and supplementary elements
        * "actif" for active elements
        * "row_sup" for supplementary rows
        * "col_sup" for supplementary columns
        * "quali_sup" for supplementary qualitative variables

    Format
    ------
    A data frame with 18 rows and 9 columns. Rows represent the different reasons mentioned, columns represent the different categories (education, age) people belong to.

    Source
    ------
    The children2 dataset from FactoMineR with supplementary qualitative variables

    Examples
    --------
    ```python
    >>> #load children2 dataset
    >>> from scientisttools import load_children2
    >>> children2 = load_children2()
    >>> res_ca = CA(row_sup=(14,15,16,17),col_sup=(5,6,7),quali_sup=8)
    >>> res_ca.fit(children2)
    ```
    """
    if element == "actif":
        return read_excel(DATASETS_DIR/"children2.xlsx",sheet_name="Feuil1",index_col=0,header=0)
    elif element == "row_sup":
        return read_excel(DATASETS_DIR/"children2.xlsx",sheet_name="Feuil2",index_col=0,header=0)
    elif element == "col_sup":
        return read_excel(DATASETS_DIR/"children2.xlsx",sheet_name="Feuil3",index_col=0,header=0)
    elif element == "quali_sup":
        return read_excel(DATASETS_DIR/"children2.xlsx",sheet_name="Feuil4",index_col=0,header=0)
    elif element == "all":
        return read_excel(DATASETS_DIR/"children2.xlsx",sheet_name="Feuil5",index_col=0,header=0)
    else:
        raise ValueError("'element' must be one of 'all', 'actif', 'row_sup', 'col_sup', 'quali_sup'")

def load_congressvotingrecords():
    """
    Congressional Voting Records
    ----------------------------

    Usage
    -----
    ```python
    >>> #load congressvotingrecords dataset
    >>> from scientisttools import load_congressvotingrecords
    >>> vote = load_congressvotingrecords()
    ```
    
    Source
    ------
    The Congressional Voting Records. See https://archive.ics.uci.edu/dataset/105/congressional+voting+records
    
    Examples
    --------
    ```python
    >>> #load congressvotingrecords dataset
    >>> from scientistools import load_congressvotingrecords
    >>> vote = load_congressvotingrecords()
    >>> from scientisttools import CATVARHCA
    >>> X = vote.iloc[:,1:]
    >>> res_catvarhca =  CATVARHCA(n_clusters=2,diss_metric="cramer",metric="euclidean",method="ward",parallelize=True)
    >>> res_catvarhca.fit(X)
    ```
    """
    return read_excel(DATASETS_DIR/"congressvotingrecords.xlsx")

def load_decathlon(element = "all"):
    """
    Performance in decathlon (data)
    -------------------------------

    Description
    -----------
    The data used here refer to athletes' performance during two sporting events.

    Usage
    -----
    ```python
    >>> #load decathlon dataset
    >>> from scientisttools import load_decathlon
    >>> decathlon = load_decathlon()
    ```

    Format
    ------
    A data frame with 46 rows and 13 columns: the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. The columns 11 and 12 correspond respectively to the rank and the points obtained. The last column is a categorical variable corresponding to the sporting event (2004 Olympic Game or 2004 Decastar)
    Supplementary individuals are the top 5 from the 1988 Seoul Olympics.

    Examples
    --------
    ```python
    >>> #load decathlon dataset
    >>> from scientisttools import load_decathlon
    >>> decathlon = load_decathlon()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(ind_sup=(41,42,43,44,45),quanti_sup=(10,11),quali_sup=12,rotation=None)
    >>> res_pca.fit(decathlon)
    ```
    """
    if element == "actif":
        return read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil1",index_col=0,header=0)
    elif element == "ind_sup":
        return read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil2",index_col=0,header=0)
    elif element == "quanti_sup":
        return read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil3",index_col=0,header=0)
    elif element == "quali_sup":
        return read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil4",index_col=0,header=0)
    elif element == "all":
        return read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil5",index_col=0,header=0)
    else:
        raise ValueError("'element' must be one of 'all', 'actif', 'ind_sup', 'quanti_sup', 'quali_sup'")

def load_femmetravail():
    """
    Femmes travail Dataset
    ----------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_femmetravail
    >>> femmetravail = load_femmetravail()
    ```

    Format
    ------
    A data frame with 3 rows and 7 columns

    Examples
    --------
    ```python
    >>> # load women_work dataset
    >>> from scientisttools import load_femmetravail
    >>> femmetravail = load_femmetravail()
    >>> from scientisttools import CA
    >>> res_ca = CA(col_sup=[3,4,5,6])
    >>> res_ca.fit(femmetravail)
    ```
    """
    return read_csv(DATASETS_DIR/"femmetravail.csv",delimiter=";",encoding =  "cp1252",index_col =0)

def load_gironde(element="all"):
    """
    Gironde Dataset
    ---------------

    Description
    -----------
    a dataset with 542 individuals and 27 columns

    Usage
    -----
    ```python
    >>> from scientisttools import load_gironde
    >>> gironde = load_gironde()
    ```

    Examples
    --------
    ```python
    >>> from scientisttools import load_gironde, PCAMIX
    >>> gironde = load_gironde()
    >>> res_pcamix = PCAMIX()
    >>> res_pcamix.fit(gironde)
    ```
    """
    if element == "employment":
        return read_r(DATASETS_DIR/"gironde_employment.rda")["employment"]
    elif element == "housing":
        return read_r(DATASETS_DIR/"gironde_housing.rda")["housing"]
    elif element == "services":
        return read_r(DATASETS_DIR/"gironde_services.rda")["services"]
    elif element == "environment":
        return read_r(DATASETS_DIR/"gironde_environment.rda")["environment"]
    elif element == "all":
        return read_r(DATASETS_DIR/"gironde.rda")["gironde"]
    else:
        Exception("'element' should be one of 'employment', 'housing', 'services', 'environment', 'all'")

def load_housetasks():
    """
    House tasks contingency table
    -----------------------------

    Description
    -----------
    A data frame containing the frequency of execution of 13 house tasks in the couple. This table is also available in ade4 R package.

    Usage
    -----
    ```python
    >>> from scientisttools import load_housetasks
    >>> housetasks = load_housetasks()
    ```

    Return
    ------
    a pandas dataframe with 13 observations (house tasks) on the following 4 columns : Wife, Alternating, Husband and Jointly

    Source
    ------
    The housetasks dataset from factoextra.See [https://rpkgs.datanovia.com/factoextra/reference/housetasks.html](https://rpkgs.datanovia.com/factoextra/reference/housetasks.html)

    Examples
    --------
    ```python
    >>> #load housetasks datasest
    >>> from scientisttools import load_housetasks
    >>> housetasks = load_housetasks()
    >>> from scientisttools import CA
    >>> res_ca = CA()
    >>> res_ca.fit(housetasks)
    ```
    """
    data = read_r(DATASETS_DIR/"housetasks.rda")["housetasks"]
    return data

def load_jobrate():
    """
    Jobrate Dataset
    ---------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_jobrate
    >>> jobrate = load_jobrate()
    ```

    Examples
    --------
    ```python
    >>> #load jobrate datasets
    >>> from scientisttools import load_jobrate
    >>> jobrate = load_jobrate()
    >>> from scientisttools import VARHCA
    >>> varhca = VARHCA(n_clusters=4,var_sup=13,matrix_type="completed",metric="euclidean",method="ward",parallelize=False)
    >>> varhca.fit(jobrate)
    ```
    """
    return read_excel(DATASETS_DIR/"jobrate.xlsx",index_col=None,header=0)

def load_lifecyclesavings():
    """
    Intercountry Life-Cycle Savings Data
    ------------------------------------

    Description
    -----------
    Data on the savings ratio 1960 - 1970

    Usage
    -----
    ```python
    >>> from scientisttools import load_lifecyclesavings
    >>> lifecyclesavings = load_lifecyclesavings()
    ```

    Format
    -----
    A data frame with 50 observations on 5 variables

    Source
    ------
    The LifeCycle Savings dataset from R datasets

    Examples
    --------
    ```python
    >>> #load lifecyclesavings dataset
    >>> from scientisttools import load_lifecyclesavings
    >>> lifecyclesavings = load_lifecyclesavings()
    >>> from scientisttools import CCA
    >>> res_cca = CCA(lifecyclesavings,vars=[1,2])
    ```  
    """
    return read_r(DATASETS_DIR/"LifeCycleSavings.RData")["LifeCycleSavings"]

def load_madagascar():
    """
    Madagascar Multidimensional Scaling Dataset
    -------------------------------------------

    Usage
    -----
    ```python
    >>> from  scientisttools import load_madagascar
    >>> madagascar = load_madagascar()

    Examples
    --------
    ```python
    >>> #load madagascar dataset
    >>> from scientisttools import load_madagascar
    >>> madagascar = load_madagascar()
    >>> from scientisttools import MDS
    >>> my_mds = MDS(n_components=None,proximity ="precomputed",normalized_stress=True)
    >>> my_mds.fit(madagascar)
    ```
    """
    return read_excel(DATASETS_DIR/"madagascar.xlsx",index_col=0,header=0)

def load_mortality():
    """
    The cause of mortality in France in 1979 and 2006
    -------------------------------------------------

    Description
    -----------
    The cause of mortality in France in 1979 and 2006

    Usage
    -----
    ```python
    >>> from scientisttools import load_mortality
    >>> mortality = load_mortality()
    ```

    Format
    ------
    A data frame with 62 rows (the different causes of death) and 18 columns. Each column corresponds to an age interval (15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85-94, 95 and more) in a year. The 9 first columns correspond to data in 1979 and the 9 last columns to data in 2006. In each cell, the counts of deaths for a cause of death in an age interval (in a year) is given.

    Source
    ------
    The mortality dataset from FactoMineR

    Examples
    --------
    ```python
    >>> #load mortality dataset
    >>> from scientisttools import load_mortality
    >>> mortality = load_mortality()
    >>> import pandas as pd
    >>> from scientisttools import MFACT
    >>> mortality2 = mortality.copy()
    >>> mortality2.columns = [x + "-2" for x in mortality2.columns]
    >>> dat = pd.concat((mortality,mortality2),axis=1)
    >>> res_mfact = MFACT(group=[9]*4,name_group=["1979","2006","1979-2","2006-2"],num_group_sup=[2,3],ind_sup=list(range(50,dat.shape[0])),parallelize=True)
    >>> res_mfact.fit(dat)
    ```
    """
    return read_r(DATASETS_DIR/"mortality.rda")["mortality"]

def load_mushroom():
    """
    Mushroom Dataset
    ----------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_mushroom
    >>> mushroom = load_mushroom()
    ```

    Source
    ------
    The Mushroom uci dataset. See https://archive.ics.uci.edu/dataset/73/mushroom

    Examples
    --------
    ```python
    >>> #load mushroom dataset
    >>> from scientisttools import load_mushroom
    >>> mushroom = load_mushroom()
    >>> from scientisttools import MCA
    >>> res_mca = MCA()
    >>> res_mca.fit(mushroom)
    ```
    """
    return read_excel(DATASETS_DIR/"mushroom.xlsx")

def load_music():
    """
    Music Dataset
    -------------

    Description
    -----------
    The data concerns tastes for music of a set of 500 individuals. It contains 5 variables of likes for music genres (french pop, rap, rock, jazz and classical), 2 variables about music listening and 2 additional variables (gender and age).

    Usage
    -----
    ```python
    >>> from scientisttools import load_music
    >>> music = load_music()
    ```
    
    Format
    ------
    a pandas DataFrame with 500 observations and 7 variables

    Source
    ------
    The Music dataset in R GDAtools packages

    Examples
    --------
    ```python
    >>> #load music dataset
    >>> from scientisttools import load_music, SpecificMCA
    >>> music = load_music()
    >>> excl = {"FrenchPop" : "NA", "Rap" : "NA" , "Rock" : "NA", "Jazz" : "NA","Classical" : "NA"}
    >>> res_spemca = SpecificMCA(n_components=5,excl=excl)
    >>> res_spemca.fit(music)
    ```
    """
    return read_r(DATASETS_DIR/"music.RData")["Music"]

def load_olympic(element = "all"):
    """
    Olympic Decathlon
    -----------------

    Description
    -----------
    This data set gives the performances of 33 men's decathlon at the Olympic Games (1988) and 5 men's decathlon at the Olympic Games (2004)

    Usage
    -----
    ```python
    >>> #load olympic dataset
    >>> from scientisttools import load_olympic
    >>> olympic = load_olympic()
    ```

    Format
    ------
    A data frame with 38 rows and 12 columns: the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. The columns 11 and 12 correspond respectively to the rank and the points obtained.
    Supplementary individuals are the top 5 from the 2004 decathlon Olympic Games.

    Examples
    --------
    ```python
    >>> #load olympic dataset
    >>> from scientisttools import load_olympic
    >>> olympic = load_olympic()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(ind_sup=(33,34,35,36,37),quanti_sup=(10,11),rotation=None)
    >>> res_pca.fit(olympic)
    ```
    """
    if element == "actif":
        return read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil1",index_col=0,header=0)
    elif element == "ind_sup":
        return read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil2",index_col=0,header=0)
    elif element == "quanti_sup":
        return read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil3",index_col=0,header=0)
    elif element == "all":
        return read_excel(DATASETS_DIR/"olympic.xlsx",sheet_name="Feuil4",index_col=0,header=0)
    else:
        raise ValueError("'element' must be one of 'all', 'actif', 'ind_sup', 'quanti_sup'")

def load_poison(element="all"):
    """
    Poison dataset
    --------------

    Description
    -----------
    The data used here refer to a survey carried out on a sample of children of primary school who suffered from food poisoning. They were asked about their symptoms and about what they ate.

    Usage
    -----
    ```python
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    ```

    Format
    ------
    A data frame with 55 rows and 15 columns.

    Source
    ------
    The poison dataset from FactoMineR

    Examples
    --------
    ```python
    >>> #load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(quali_sup=(2,3),quanti_sup=(0,1))
    >>> res_mca.fit(poison)
    ```
    """
    if element == "actif":
        return read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil1",index_col=0,header=0)
    elif element == "quali_sup":
        return read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil2",index_col=0,header=0)
    elif element == "quanti_sup":
        return read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil3",index_col=0,header=0)
    elif element == "all":
        return read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil4",index_col=0,header=0)
    else:
        raise ValueError("'element' must be one of 'all', 'actif', 'quali_sup', 'quanti_sup'")

def load_protein():
    """
    Protein dataset
    ---------------

    Description
    -----------
    This dataset gives the amount of protein consumed for nine food groups in 25 European countries. The nine food groups are red meat (RedMeat), white meat (WhiteMeat), eggs (Eggs), milk (Milk), fish (Fish), cereal (Cereal), starch (Starch), nuts (Nuts), and fruits and vegetables (FruitVeg).

    Usage
    -----
    ```python
    >>> from scientisttools import load_protein
    >>> protein = load_protein()
    ```

    Format
    ------
    A numerical data matrix with 25 rows (the European countries) and 9 columns (the food groups)

    Source
    ------
    The protein dataset for sparsePCA R package
    """
    return read_r(DATASETS_DIR/"protein.RData")["protein"]

def load_qtevie():
    """
    Qualité de vie dataset
    ----------------------

    Description
    -----------
    34 country of OCDE with Russia and Brazil describe by 22 indicators and one qualitative variable group by theme : 
        * material.well.being (5), 
        * employment (5), 
        * satisfaction (3), 
        * health.and.safety (6), 
        * education (3)
        * region (1)

    Usage
    -----
    ```python
    >>> from scientisttools import load_qtevie
    >>> qtevie = load_qtevie()
    ```
    Source
    ------
    OCDE

    Examples
    --------
    ```python
    >>> from scientisttools import load_qtevie
    >>> qtevie = load_qtevie()
    >>> from scientisttools import MFA
    >>> name = ["material.well.being","employment","satisfaction","health.and.safety","education","region"] 
    >>> group_type = ["s","s","s","s","s","n"]
    >>> res_mfa = MFA(n_components=5,group=[5,5,3,6,3,1],name_group=name,group_type=group_type,num_group_sup=5)
    >>> res_mfa.fit(qtevie)
    ```
    """
    return read_csv(DATASETS_DIR/"qtevie.csv",encoding="ISO-8859-1",header=0,sep=";",index_col=0)

def load_racescanines(element="all"):
    """
    Races canines Dataset
    ---------------------

    Description
    -----------
    The data contains 32 individuals

    Usage
    -----
    ```python
    >>> from scientisttools import load_racescanines
    >>> canines = load_racescanines()
    ```
    
    Examples
    --------
    ```python
    >>> #load racescanines dataset
    >>> from scientisttools import load_racescanines
    >>> canines = load_racescanines()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(ind_sup=(27,28,29,30,31),quali_sup=6,quanti_sup=7)
    >>> res_mca.fit(canines)
    ```
    """
    if element == "actif":
         return read_excel(DATASETS_DIR/"racescanines.xlsx",sheet_name="Feuil1",header=0,index_col=0)
    elif element == "ind_sup":
         return read_excel(DATASETS_DIR/"racescanines.xlsx",sheet_name="Feuil2",header=0,index_col=0)
    elif element == "quali_sup":
         return read_excel(DATASETS_DIR/"racescanines.xlsx",sheet_name="Feuil3",header=0,index_col=0)
    elif element == "quanti_sup":
         return read_excel(DATASETS_DIR/"racescanines.xlsx",sheet_name="Feuil4",header=0,index_col=0)
    elif element == "all":
         return read_excel(DATASETS_DIR/"racescanines.xlsx",sheet_name="Feuil5",header=0,index_col=0)
    else:
        Exception("'element' must be one of 'all', 'actif', 'ind_sup', 'quali_sup', 'quanti_sup'")

def load_tea():
    """
    Tea Dataset
    -----------

    Description
    -----------
    The data used here concern a questionnaire on tea. We asked to 300 individuals how they drink tea (18 questions), what are their product's perception (12 questions) and some personal details (4 questions).

    Usage
    -----
    ```python
    >>> from scientisttools load_tea
    >>> tea = load_tea()
    ```

    Return
    ------
    A pandas DataFrame with 300 rows and 36 columns. Rows represent the individuals, columns represent the different questions. The first 18 questions are active ones, the 19th is a supplementary quantitative variable (the age) and the last variables are supplementary categorical variables.

    Source
    ------
    The tea dataset from FactoMineR.

    Examples
    --------
    ```python
    >>> #load tea dataset
    >>> from scientisttools import load_tea
    >>> tea = load_tea()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(quanti_sup=18,quali_sup=list(range(19,36)))
    >>> res_mca.fit(tea)
    ```
    """
    return read_r(DATASETS_DIR/"tea.rda")["tea"]

def load_temperature(element="all"):
    """
    Temperature Dataset
    -------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_temperature
    >>> temperature = load_temperature()
    ```

    Examples
    --------
    ```python
    >>> #load temperature dataset
    >>> from scientisttools import load_temperature
    >>> temperature = load_temperature()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(ind_sup=(15,16,17,18,19,20,21,22,23,24),quanti_sup=(12,13,14,15),quali_sup=16,rotation=None)
    >>> res_pca.fit(temperature)
    ```
    """
    if element == "actif":
         return read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil1",header=0,index_col=0)
    elif element == "ind_sup":
         return read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil2",header=0,index_col=0)
    elif element == "quanti_sup":
         return read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil3",header=0,index_col=0)
    elif element == "quali_sup":
         return read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil4",header=0,index_col=0)
    elif element == "all":
         return read_excel(DATASETS_DIR/"temperature.xlsx",sheet_name="Feuil5",header=0,index_col=0)
    else:
        Exception("'element' must be one of 'all', 'actif', 'ind_sup', 'quanti_sup', 'quali_sup'")

def load_tennis(element="all"):
    """
    Tennis 2020 Dataset
    -------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_tennis
    >>> tennis2020 = load_tennis()
    ```
    
    Examples
    --------
    ```python
    >>> # Load tennis dataset
    >>> from scientisttools import load_tennis
    >>> tennis = load_tennis()
    >>> from scientisttools import FAMD
    >>> res_famd =  FAMD(n_components=2,ind_sup=(16,17,18,19),quanti_sup=7)
    >>> res_famd.fit(tennis)
    ```
    """
    if element == "actif":
         return read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil1",header=0,index_col=0)
    elif element == "ind_sup":
         return read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil2",header=0,index_col=0)
    elif element == "quanti_sup":
         return read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil3",header=0,index_col=0)
    elif element == "all":
         return read_excel(DATASETS_DIR/"tennis.xlsx",sheet_name="Feuil4",header=0,index_col=0)
    else:
        Exception("'element' must be one of 'all', 'actif', 'ind_sup', 'quanti_sup'")

def load_usarrests():
    """
    Violent Crime Rates by US State
    -------------------------------

    Description
    -----------
    This data set contains statistics, in arrests per 100,000 residents for assault, murder, and rape in each of the 50 US states in 1973. Also given is the percent of the population living in urban areas.

    Usage
    -----
    ```python
    >>> from scientisttools import load_usarrests
    >>> usarrests = load_usarrests()
    ```
    
    Format
    ------
    dataframe with 50 observations on 4 variables.

    `Murder`:	numeric	Murder arrests (per 100,000)
    
    `Assault`: 	numeric	Assault arrests (per 100,000)
    
    `UrbanPop`: numeric Percent urban population
    
    `Rape`: numeric Rape arrests (per 100,000)

    Source
    ------
    World Almanac and Book of facts 1975. (Crime rates).

    Statistical Abstracts of the United States 1975, p.20, (Urban rates), possibly available as https://books.google.ch/books?id=zl9qAAAAMAAJ&pg=PA20.

    References
    ----------
    McNeil, D. R. (1977) Interactive Data Analysis. New York: Wiley.
    """
    return read_excel(DATASETS_DIR/"usarrests.xlsx",index_col=0,header=0)

def load_uscrime():
    """
    US Crime Dataset
    ----------------

    Description
    -----------
    These data are crime-related and demographic statistics for 47 US states in 1960. The data were collected from the FBI's Uniform Crime Report and other government agencies to determine how the variable crime rate depends on the other variables measured in the study.
    
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

    Usage
    -----
    ```python
    >>> #load uscrime dataset
    >>> from scientisttools import load_uscrime
    >>> uscrime = load_uscrime()
    ```
    
    Source
    ------
    see https://lib.stat.cmu.edu/DASL/Datafiles/USCrime.html

    Reference
    ---------
    Vandaele, W. (1978) Participation in illegitimate activities: Erlich revisited. In Deterrence and incapacitation, Blumstein, A., Cohen, J. and Nagin, D., eds., Washington, D.C.: National Academy of Sciences, 270-335. Methods: A Primer, New York: Chapman & Hall, 11. Also found in: Hand, D.J., et al. (1994) A Handbook of Small Data Sets, London: Chapman & Hall, 101-103.

    Examples
    --------
    ```python
    >>> #load uscrime dataset
    >>> from scientistools import load_uscrime
    >>> uscrime = load_uscrime()
    >>> from scientisttools import FAMD
    >>> res_famd = FAMD()
    >>> res_famd.fit(uscrime)
    ```
    """
    return read_excel(DATASETS_DIR/"crime.xlsx",sheet_name="Feuil1",header=0,index_col=0)

def load_wine():
    """
    Wine dataset
    ------------

    Description
    -----------
    The data used here refer to 21 wines of Val de Loire

    Usage
    -----
    ```python
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    ```

    Format
    ------
    A data frame with 21 rows (the number of wines) and 31 columns: 
        * the first column corresponds to the label of origin, 
        * the second column corresponds to the soil, 
        * and the others correspond to sensory descriptors.
    
    Source
    ------
    The wine dataset from FactoMineR

    Examples
    --------
    ```python
    >>> # Load wine data
    >>> from scientisttools import load_wine
    >>> wine = load_wine()
    >>> # Example of PCA
    >>> from scientisttools import PCA
    >>> res_pca = PCA(standardize=True,n_components=5,quanti_sup=[29,30],quali_sup=[0,1],parallelize=True)
    >>> res_pca.fit(wine) 
    >>> # Example of MCA
    >>> from scientisttools import MCA
    >>> res_mca = MCA(quanti_sup = list(range(2,wine.shape[1])))
    >>> res_mca.fit(wine)
    >>> # Example of FAMD
    >>> from scientisttools import FAMD
    >>> res_famd = FAMD()
    >>> res_famd.fit(wine)
    >>> # Example of MFA
    >>> from scientisttools import MFA
    >>> res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    >>> res_mfa.fit(wine)
    ```
    """
    return read_r(DATASETS_DIR/"wine.rda")["wine"]

def load_womenwork():
    """
    Women Work Dataset
    ------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_womenwork
    >>> womenwork = load_womenwork()
    ```

    Format
    ------
    A pandas DataFrame with 3 rows and 7 columns

    Examples
    --------
    ```python
    >>> # load womenwork dataset
    >>> from scientisttools import load_women_work
    >>> womenwork = load_womenwork()
    >>> from scientisttools import CA
    >>> res_ca = CA(col_sup=(3,4,5,6))
    >>> res_ca.fit(women_work)
    ```
    """
    return read_csv(DATASETS_DIR/"womenwork.txt",sep="\t")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Datasets as DataFrame
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

autos1990 = read_csv(DATASETS_DIR/'autos1990.txt', delimiter = " ",header=0,index_col=0)
decathlon = read_excel(DATASETS_DIR/"decathlon.xlsx",sheet_name="Feuil5",index_col=0,header=0)
poison = read_excel(DATASETS_DIR/"poison.xlsx",sheet_name="Feuil4",index_col=0,header=0)
racescanines = read_excel(DATASETS_DIR/"racescanines.xlsx",sheet_name="Feuil5",header=0,index_col=0)