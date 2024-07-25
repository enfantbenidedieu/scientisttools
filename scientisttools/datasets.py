# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pyreadr 
import pathlib

# https://husson.github.io/data.html
# https://r-stat-sc-donnees.github.io/liste_don.html

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"

def load_autos():
    """
    Autos 2005 - Données sur 40 voitures
    -------------------------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_autos
    >>> autos = load_autos()
    ```
    
    Examples
    --------
    ```python
    >>> # Load autos dataset
    >>> from scientisttools import load_autos
    >>> autos = load_autos()
    >>> # Example of PCA
    >>> res_pca = PCA(quanti_sup=[10,11],quali_sup = [12,13,14])
    >>> res_pca.fit(autos)
    >>> # Example of FAMD
    >>> res_afdm = FAMD(quanti_sup=[10,11],quali_sup=14,parallelize=False)
    >>> res_afdm.fit(autos)
    ```
    """
    data = pd.read_excel(DATASETS_DIR/"autos2005.xls",header=0,index_col=0)
    return data

def load_autos2():
    """
    Autos Data - Données sur 45 voitures
    -----------------------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_autos2005
    >>> auto2 = laod_autos2()
    ```

    Examples
    --------
    ```python
    >>> # Load dataset
    >>> from scientisttools import load_autos2
    >>> autos2 = load_autos2()
    >>> from scientisttools import FAMD
    >>> res_famd = FAMD(ind_sup=list(range(38,autos2.shape[0])),quanti_sup=[12,13,14],quali_sup=15)
    >>> res_famd.fit(autos2)
    ```
    """
    data = pd.read_excel(DATASETS_DIR/"autos2005_afdm.xlsx",header=0,index_col=0)
    return data

def load_autosmds():
    """
    Autos Multidimensional Scaling dataset
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
    >>> # Load autosmds dataset
    >>> from scientisttools import load_autosmds
    >>> autosmds = load_autosmds()
    >>> from scientisttools import CMDSCALE
    >>> my_cmds = CMDSCALE(n_components=2,ind_sup=[12,13,14],proximity="euclidean",normalized_stress=True,parallelize=False)
    >>> my_cmds.fit(autosmds)
    ```
    """
    autosmds = pd.read_excel(DATASETS_DIR/"autosmds.xlsx",index_col=0,header=0)
    autosmds.index.name = None
    return autosmds

def load_body():
    """
    Body dimensions datasets
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
    body = pd.read_excel(DATASETS_DIR/"body.xls",sheet_name="body")
    return body

def load_burgundywines():
    """
    Burgundy wines dataset
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
    wines = pd.DataFrame(
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

def load_cars2006(which="actif"):
    """
    Cars dataset
    ------------

    Description
    -----------
    18 cars described by 6 quantitatives variables

    Usage
    -----
    ```python
    >>> from scientisttools import load_cars2006
    >>> cars = load_cars2006()
    ```

    Parameters
    ----------
    which : the element to subset from the output. Allowed values are :
        * "actif" for actifs elements
        * "indsup" for supplementary individuals
        * "varquantsup" for supplementary quantitatives variables
        * "varqualsup" for supplementary qualitatives variables
    
    Returns
    -------
    pandas dataframe

    References
    ----------
    Saporta G. (2006). Probabilites, Analyse des données et Statistiques. Technip

    Rakotomalala R. (2020). Pratique des méthodes factorielles avec Python. Université Lumière Lyon 2. Version 1.0

    Examples
    --------
    ```python
    >>> # load cars2006 dataset
    >>> from scientisttools import load_cars2006
    >>> D = load_cars2006(which="actif")
    >>> from scientisttools import PCA
    >>> res_pca = PCA(n_components=5)
    >>> res_pca.fit(D)
    >>> # Load supplementary individuals
    >>> ind_sup = load_cars2006(which="indsup")
    >>> ind_sup_coord = res_pca.transform(ind_sup)
    ```
    """
    if which not in ["actif","indsup","varquantsup","varqualsup"]:
        raise ValueError("'which' must be one of 'actif', 'indsup', 'varquantsup', 'varqualsup'")

    if which == "actif":
        cars = pd.read_excel(DATASETS_DIR/"cars2006.xlsx",sheet_name="actif",index_col=0,header=0)
    elif which == "indsup":
        cars = pd.read_excel(DATASETS_DIR/"cars2006.xlsx",sheet_name="ind. sup.",index_col=0,header=0)
    elif which == "varquantsup":
        cars = pd.read_excel(DATASETS_DIR/"cars2006.xlsx",sheet_name="var. illus. quant.",index_col=0,header=0)
    elif which == "varqualsup":
        cars = pd.read_excel(DATASETS_DIR/"cars2006.xlsx",sheet_name="var. illus. qual.",index_col=0,header=0)
    return cars

def load_carsacpm():
    """
    Cars
    ----

    Usage
    -----
    ```python
    >>> from scientisttools import load_carsacpm
    >>> cars = load_carsacpm()
    ```

    Format
    ------
    a data frame with 27 individuals and 9 variables

    Examples
    --------
    ```python
    >>> # Load cars dataset
    >>> from scientisttools import load_carsacpm
    >>> cars = load_carsacpm() 
    >>> from scientisttools import MPCA
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(cars)
    ```
    """
    data = pd.read_csv(DATASETS_DIR/'carsacpm.txt', delimiter = " ",header=0,index_col=0)
    return data

def load_children():
    """
    Children dataset
    ----------------

    Description
    -----------
    The data used here is a contingency table that summarizes the answers given by different categories of people to the following question : according to you, what are the reasons that can make hesitate a woman or a couple to have children?

    Usage
    -----
    ```python
    >>> from scientisttools import load_children
    >>> children = load_children()
    ```

    Format
    ------
    A data frame with 18 rows and 8 columns. Rows represent the different reasons mentioned, columns represent the different categories (education, age) people belong to.

    Source
    ------
    The children dataset from FactoMineR.

    Examples
    --------
    ```python
    >>> # Load children dataset
    >>> from scientisttools import load_children
    >>> children = load_children()
    >>> res_ca = CA(row_sup=list(range(14,18)),col_sup=list(range(5,8)),parallelize=True)
    >>> res_ca.fit(children)
    ```
    """
    data = pyreadr.read_r(DATASETS_DIR/"children.rda")["children"]
    return data

def load_congressvotingrecords():
    """
    Congressional Voting Records
    ----------------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_congressvotingrecords
    >>> vote = load_congressvotingrecords()
    ```
    
    Source
    ------
    The Congressional Voting Records. See https://archive.ics.uci.edu/dataset/105/congressional+voting+records
    
    Examples
    --------
    ```python
    >>> # Load vote dataset
    >>> from scientistools import load_congressvotingrecords
    >>> vote = load_congressvotingrecords()
    >>> from scientisttools import CATVARHCA
    >>> X = vote.iloc[:,1:]
    >>> res_catvarhca =  CATVARHCA(n_clusters=2,diss_metric="cramer",metric="euclidean",method="ward",parallelize=True)
    >>> res_catvarhca.fit(X)
    ```
    """
    vote = pd.read_excel(DATASETS_DIR/"congressvotingrecords.xlsx")
    return vote

def load_decathlon():
    """
    Performance in decathlon (data)
    -------------------------------

    Description
    -----------
    The data used here refer to athletes' performance during two sporting events.

    Usage
    -----
    ```python
    >>> from scientisttools import load_decathlon
    >>> decathlon = load_decathlon()
    ```

    Format
    ------
    A data frame with 41 rows and 13 columns: the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. The columns 11 and 12 correspond respectively to the rank and the points obtained. The last column is a categorical variable corresponding to the sporting event (2004 Olympic Game or 2004 Decastar)

    Source
    ------
    The decathlon dataset from FactoMineR. See [https://rdrr.io/cran/FactoMineR/man/decathlon.html](https://rdrr.io/cran/FactoMineR/man/decathlon.html)

    Examples
    --------
    ```python
    >>> # Load decathlon dataset
    >>> from scientisttools import load_decathlon
    >>> decathlon = load_decathlon()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(standardize=True,ind_sup=list(range(23,decathlon.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(decathlon)
    ```
    """
    data = pyreadr.read_r(DATASETS_DIR/"decathlon.rda")["decathlon"]
    return data

def load_decathlon2():
    """
    Athletes' performance in decathlon
    ----------------------------------

    Description
    -----------
    Athletes' performance during two sporting meetings

    Usage
    -----
    ```python
    >>> from scientisttools.datasets import load_decathlon2
    >>> decathlon2 = load_decathlon2()
    ```
    
    Format
    ------
    A data frame with 27 observations and 13 variables.

    Source
    ------
    The decathlon2 dataset from factoextra. See [https://rpkgs.datanovia.com/factoextra/reference/decathlon2.html](https://rpkgs.datanovia.com/factoextra/reference/decathlon2.html)

    Examples
    --------
    ```python
    >>> # load decathlon2 dataset
    >>> from scientisttools import load_decathlon2
    >>> decathlon2 = load_decathlon2()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(standardize=True,ind_sup=list(range(23,decathlon2.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    >>> res_pca.fit(decathlon2)
    ```
    """
    data = pyreadr.read_r(DATASETS_DIR/"decathlon2.rda")["decathlon2"]
    return data

def load_femmetravail():
    """
    Femmes travail dataset
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
    data = pd.read_csv(DATASETS_DIR/"femme_travail.csv",delimiter=";",encoding =  "cp1252",index_col =0)
    return data

def load_gironde(which="all"):
    """
    gironde dataset
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
    if which not in ["employment","housing","services","environment","all"]:
        raise ValueError("'which' should be one of 'employment', 'housing', 'services', 'environment', 'all'")
    
    if which == "employment":
        gironde = pyreadr.read_r(DATASETS_DIR/"gironde_employment.rda")["employment"]
    elif which == "housing":
        gironde = pyreadr.read_r(DATASETS_DIR/"gironde_housing.rda")["housing"]
    elif which == "services":
        gironde = pyreadr.read_r(DATASETS_DIR/"gironde_services.rda")["services"]
    elif which == "environment":
        gironde = pyreadr.read_r(DATASETS_DIR/"gironde_environment.rda")["environment"]
    else:
        gironde = pyreadr.read_r(DATASETS_DIR/"gironde.rda")["gironde"]
    return gironde

def load_housetasks():
    """
    House tasks contingency table
    ----------------------------

    Description
    -----------
    A data frame containing the frequency of execution of 13 house tasks in the couple. This table is also available in ade4 R package.

    Usage
    -----
    ```python
    >>> from scientisttools import load_housetasks
    >>> housetasks = load_housetasks()
    ```

    Format
    ------
    dataframe with 13 observations (house tasks) on the following 4 columns : Wife, Alternating, Husband and Jointly

    Source
    ------
    The housetasks dataset from factoextra.See [https://rpkgs.datanovia.com/factoextra/reference/housetasks.html](https://rpkgs.datanovia.com/factoextra/reference/housetasks.html)

    Examples
    --------
    ```python
    >>> # Load housetasks datasest
    >>> from scientisttools import load_housetasks
    >>> housetasks = load_housetasks()
    >>> from scientisttools import CA
    >>> res_ca = CA()
    >>> res_ca.fit(housetasks)
    ```
    """
    data = pyreadr.read_r(DATASETS_DIR/"housetasks.rda")["housetasks"]
    return data

def load_jobrate():
    """
    Jobrate dataset
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
    >>> # Load jobrate
    >>> from scientisttools import load_jobrate
    >>> jobrate = load_jobrate()
    >>> from scientisttools import VARHCA
    >>> varhca = VARHCA(n_clusters=4,var_sup=13,matrix_type="completed",metric="euclidean",method="ward",parallelize=False)
    >>> varhca.fit(jobrate)
    ```
    """
    jobrate = pd.read_excel(DATASETS_DIR/"jobrate.xlsx",index_col=None,header=0)
    jobrate.index.name = None
    return jobrate

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
    >>> # Load lifecyclesavings dataset
    >>> from scientisttools import load_lifecyclesavings
    >>> lifecyclesavings = load_lifecyclesavings()
    >>> from scientisttools import CCA
    >>> res_cca = CCA(lifecyclesavings,vars=[1,2])
    ```  
    """
    data = pyreadr.read_r(DATASETS_DIR/"LifeCycleSavings.RData")["LifeCycleSavings"]
    return data

def load_madagascar():
    """
    Madagascar Multidimensional Scaling dataset
    -------------------------------------------

    Usage
    -----
    ```python
    >>> from  scientisttools import load_madagascar
    >>> madagascar = load_madagascar()

    Examples
    --------
    ```python
    >>> # Load dataset
    >>> from scientisttools import load_madagascar
    >>> madagascar = load_madagascar()
    >>> from scientisttools import MDS
    >>> my_mds = MDS(n_components=None,proximity ="precomputed",normalized_stress=True)
    >>> my_mds.fit(madagascar)
    ```
    """
    madagascar = pd.read_excel(DATASETS_DIR/"madagascar.xlsx",index_col=0,header=0)
    return madagascar

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
    >>> # load mortality dataset
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
    data = pyreadr.read_r(DATASETS_DIR/"mortality.rda")["mortality"]
    return data

def load_mushroom():
    """
    Mushroom dataset
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
    >>> # Load mushroom dataset
    >>> from scientisttools import load_mushroom
    >>> mushroom = load_mushroom()
    >>> from scientisttools import MCA
    >>> res_mca = MCA()
    >>> res_mca.fit(mushroom)
    ```
    """
    data = pd.read_excel(DATASETS_DIR/"mushroom.xlsx")
    return data

def load_music():
    """
    Music dataset
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
    A data frame with 500 observations and 7 variables

    Source
    ------
    The Music dataset in R GDAtools packages

    Examples
    --------
    ```python
    >>> # Load music dataset
    >>> from scientisttools import load_music, SpecificMCA
    >>> music = load_music()
    >>> excl = {"FrenchPop" : "NA", "Rap" : "NA" , "Rock" : "NA", "Jazz" : "NA","Classical" : "NA"}
    >>> res_spemca = SpecificMCA(n_components=5,excl=excl)
    >>> res_spemca.fit(music)
    ```
    """
    data = pyreadr.read_r(DATASETS_DIR/"Music.RData")["Music"]
    return data

def load_poison():
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
    >>> # Load poison dataset
    >>> from scientisttools import load_poison
    >>> poison = load_poison()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1])
    >>> res_mca.fit(poison)
    ```
    """
    data = pyreadr.read_r(DATASETS_DIR/"poison.rda")["poison"]
    return data

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
    data = pyreadr.read_r(DATASETS_DIR/"protein.RData")["protein"]
    return data

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
    data = pd.read_csv(DATASETS_DIR/"QteVie.csv",encoding="ISO-8859-1",header=0,sep=";",index_col=0)
    return data

def load_racescanines():
    """
    Races canines dataset
    ---------------------

    Description
    -----------
    The data contains 27 individuals

    Usage
    -----
    ```python
    >>> from scientisttools import load_racescanines
    >>> canines = load_racescanines()
    ```
    
    Examples
    --------
    ```python
    >>> # Load races canines dataset
    >>> from scientisttools import load_races_canines
    >>> canines = load_races_canines()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(ind_sup=list(range(27,races_canines.shape[0])),quanti_sup=7,quali_sup=6)
    >>> res_mca.fit(canines)
    ```
    """
    data = pd.read_excel(DATASETS_DIR/"races_canines.xlsx",header=0,index_col=0)
    return data

def load_tea():
    """
    tea dataset
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

    Format
    ------
    A data frame with 300 rows and 36 columns. Rows represent the individuals, columns represent the different questions. The first 18 questions are active ones, the 19th is a supplementary quantitative variable (the age) and the last variables are supplementary categorical variables.

    Source
    ------
    The tea dataset from FactoMineR.

    Examples
    --------
    ```python
    >>> # Load tea dataset
    >>> from scientisttools import load_tea
    >>> tea = load_tea()
    >>> from scientisttools import MCA
    >>> res_mca = MCA(quanti_sup=18, quali_sup=list(range(19,36)))
    >>> res_mca.fit(tea)
    ```
    """
    data = pyreadr.read_r(DATASETS_DIR/"tea.rda")["tea"]
    return data

def load_temperature():
    """
    Temperature dataset
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
    >>> # Load temperature dataset
    >>> from scientisttools import load_temperature
    >>> temperature = load_temperature()
    >>> from scientisttools import PCA
    >>> res_pca = PCA(ind_sup=list(range(15,temperatuer.shape[0])),quanti_sup=list(range(12,16)),quali_sup=16)
    >>> res_pca.fit(temperature)
    ```
    """
    data = pd.read_excel(DATASETS_DIR/"temperature.xlsx",header=0,index_col=0)
    return data

def load_tennis():
    """
    Tennis dataset
    --------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_tennis
    >>> tennis = load_tennis()
    ```
    
    Examples
    --------
    ```python
    >>> # Load tennis dataset
    >>> from scientisttools import load_tennis
    >>> tennis = load_tennis()
    >>> from scientisttools import FAMD
    >>> res_famd =  FAMD(n_components=2,ind_sup=list(range(16,tennis.shape[0])),quanti_sup=7)
    >>> res_famd.fit(tennis)
    ```
    """
    data = pd.read_excel(DATASETS_DIR/"tennisplayers.xlsx",index_col=0)
    return data

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

    `Murder` :	numeric	Murder arrests (per 100,000)
    
    `Assault`: 	numeric	Assault arrests (per 100,000)
    
    `UrbanPop` : numeric Percent urban population
    
    `Rape` : numeric Rape arrests (per 100,000)

    Source
    ------
    World Almanac and Book of facts 1975. (Crime rates).

    Statistical Abstracts of the United States 1975, p.20, (Urban rates), possibly available as https://books.google.ch/books?id=zl9qAAAAMAAJ&pg=PA20.

    References
    ----------
    McNeil, D. R. (1977) Interactive Data Analysis. New York: Wiley.
    """
    usarrests = pd.read_excel(DATASETS_DIR/"usarrests.xlsx",index_col=0,header=0)
    return usarrests

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
    data = pyreadr.read_r(DATASETS_DIR/"wine.rda")["wine"]
    return data

def load_womenwork():
    """
    Women work dataset
    ------------------

    Usage
    -----
    ```python
    >>> from scientisttools import load_womenwork
    >>> womenwork = load_womenwork()
    ```

    Format
    ------
    A data frame with 3 rows and 7 columns

    Examples
    --------
    ```python
    >>> # load women_work dataset
    >>> from scientisttools import load_women_work
    >>> women_work = load_women_work()
    >>> from scientisttools import CA
    >>> res_ca = CA(col_sup=[3,4,5,6])
    >>> res_ca.fit(women_work)
    ```
    """
    data = pd.read_csv(DATASETS_DIR/"women_work.txt",sep="\t")
    return data