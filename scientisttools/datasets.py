# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pyreadr 
import pathlib

# https://husson.github.io/data.html
# https://r-stat-sc-donnees.github.io/liste_don.html

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"

def load_cars():
    """
    Cars
    ----

    Usage
    -----
    ```python
    >>> from scientisttools import load_cars
    >>> cars = load_cars()
    ```

    Format
    ------
    a data frame with 27 individuals and 9 variables

    Examples
    --------
    ```python
    >>> # Load cars dataset
    >>> from scientisttools import load_cars
    >>> cars = load_cars() 
    >>> from scientisttools import MPCA
    >>> res_mpca = MPCA()
    >>> res_mpca.fit(cars)
    ```
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    data = pd.read_csv(DATASETS_DIR/'acpm_cars.txt', delimiter = " ",header=0,index_col=0)
    return data

def load_autos():
    """
    Autos 2005 - Données sur 40 voitures
    -------------------------------------

    Usage
    -----
    > from scientisttools import load_autos
    > autos = load_autos()

    Examples
    --------
    > # Load autos dataset
    > from scientisttools import load_autos
    > autos = load_autos()
    >
    > # Example of PCA
    > res_pca = PCA(quanti_sup=[10,11],quali_sup = [12,13,14])
    > res_pca.fit(autos)
    >
    > # Example of FAMD
    > res_afdm = FAMD(quanti_sup=[10,11],quali_sup=14,parallelize=False)
    > res_afdm.fit(autos)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    data = pd.read_excel(DATASETS_DIR/"autos2005.xls",header=0,index_col=0)
    data.name = "autos2005"
    return data

def load_autos2():
    """
    FAMD Data - Données sur 45 voitures
    -----------------------------------

    Usage
    -----
    > from scientisttools import load_autos2005
    > auto2 = laod_autos2()

    Examples
    --------
    > # Load dataset
    > from scientisttools import load_autos2
    > autos2 = load_autos2()
    > 
    > from scientisttools import FAMD
    > res_famd = FAMD(ind_sup=list(range(38,autos2.shape[0])),quanti_sup=[12,13,14],quali_sup=15)
    > res_famd.fit(autos2)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    data = pd.read_excel(DATASETS_DIR/"autos2005_afdm.xlsx",header=0,index_col=0)
    data.name = "autos_2005"
    return data

def load_cars2006(which="actif"):
    """
    Cars dataset
    ------------

    Description
    -----------
    18 cars described by 6 quantitatives variables

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

def load_decathlon():
    """
    Performance in decathlon (data)
    -------------------------------

    Description
    -----------
    The data used here refer to athletes' performance during two sporting events.

    Usage
    -----
    > from scientisttools import load_decathlon
    > decathlon = load_decathlon()

    Format
    ------
    A data frame with 41 rows and 13 columns: the first ten columns corresponds to the performance of the athletes for the 10 events of the decathlon. The columns 11 and 12 correspond respectively to the rank and the points obtained. The last column is a categorical variable corresponding to the sporting event (2004 Olympic Game or 2004 Decastar)

    Source
    ------
    The decathlon dataset from FactoMineR. See [https://rdrr.io/cran/FactoMineR/man/decathlon.html](https://rdrr.io/cran/FactoMineR/man/decathlon.html)

    Examples
    --------
    > # Load decathlon dataset
    > from scientisttools import load_decathlon
    > decathlon = load_decathlon()
    >
    > from scientisttools import PCA
    > res_pca = PCA(standardize=True,ind_sup=list(range(23,decathlon.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    > res_pca.fit(decathlon)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"decathlon.rda")["decathlon"]
    data.name = "decathlon"
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
    > from scientisttools.datasets import load_decathlon2
    > decathlon2 = load_decathlon2()

    Format
    ------
    A data frame with 27 observations and 13 variables.

    Source
    ------
    The decathlon2 dataset from factoextra. See [https://rpkgs.datanovia.com/factoextra/reference/decathlon2.html](https://rpkgs.datanovia.com/factoextra/reference/decathlon2.html)

    Examples
    --------
    > # load decathlon2 dataset
    > from scientisttools import load_decathlon2
    > decathlon2 = load_decathlon2()
    >
    > from scientisttools import PCA
    > res_pca = PCA(standardize=True,ind_sup=list(range(23,decathlon2.shape[0])),quanti_sup=[10,11],quali_sup=12,parallelize=True)
    > res_pca.fit(decathlon2)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"decathlon2.rda")["decathlon2"]
    data.name = "decathlon2"
    return data

def load_temperature():
    """
    Temperature
    -----------


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

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    data = pd.read_excel(DATASETS_DIR/"temperature.xlsx",header=0,index_col=0)
    data.name = "temperature"
    return data

############################################# Correspondance Analysis ########################################""

def load_women_work():
    """
    Women work
    ----------

    Usage
    -----
    > from scientisttools import load_women_work
    > women_work = load_women_work()

    Format
    ------
    A data frame with 3 rows and 7 columns

    Examples
    --------
    > # load women_work dataset
    > from scientisttools import load_women_work
    > women_work = load_women_work()
    > from scientisttools import CA
    > res_ca = CA(col_sup=[3,4,5,6])
    > res_ca.fit(women_work)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pd.read_csv(DATASETS_DIR/"women_work.txt",sep="\t")
    data.name = "women_work"
    return data

def load_femmes_travail():
    """
    Femmes travail
    --------------

    Usage
    -----
    > from scientisttools import load_femmes_travail
    > femmes_travail = load_femmes_travail()

    Format
    ------
    A data frame with 3 rows and 7 columns

    Examples
    --------
    > # load women_work dataset
    > from scientisttools import load_femmes_travail
    > femmes_travail = load_femmes_travail()
    > from scientisttools import CA
    > res_ca = CA(col_sup=[3,4,5,6])
    > res_ca.fit(femmes_travail)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pd.read_csv(DATASETS_DIR/"femme_travail.csv",delimiter=";",encoding =  "cp1252",index_col =0)
    data.name = "femmes_travail"
    return data

def load_body():
    """
    Body dimensions datasets
    ------------------------

    Description
    -----------
    The data give some body dimension measurements as well as age, weight, height, and gender on 507 individuals. The 247 men and 260 women were primarily individuals in their twenties and thirties, with a scattering of older men and women, all exercising serveral hours a week. 

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
    """
    body = pd.read_excel(DATASETS_DIR/"body.xls",sheet_name="body")
    return body

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

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"housetasks.rda")["housetasks"]
    data.name = "housetasks"
    return data

def load_children():
    """
    Children
    --------

    Description
    -----------
    The data used here is a contingency table that summarizes the answers given by different categories of people to the following question : according to you, what are the reasons that can make hesitate a woman or a couple to have children?

    Usage
    -----
    > from scientisttools import load_children
    > children = load_children()

    Format
    ------
    A data frame with 18 rows and 8 columns. Rows represent the different reasons mentioned, columns represent the different categories (education, age) people belong to.

    Source
    ------
    The children dataset from FactoMineR.

    Examples
    --------
    > # Load children dataset
    > from scientisttools import load_children
    > children = load_children()
    > res_ca = CA(row_sup=list(range(14,18)),col_sup=list(range(5,8)),parallelize=True)
    > res_ca.fit(children)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"children.rda")["children"]
    data.name = "children"
    return data

def load_races_canines():
    """
    Races canines
    -------------

    Description
    -----------
    The data contains 27 individuals

    Usage
    -----
    > from scientisttools import load_races_canines
    > races_canines = load_races_canines()

    Examples
    --------
    > # Load races canines dataset
    > from scientisttools import load_races_canines
    > races_canines = load_races_canines()
    >
    > from scientisttools import MCA
    > res_mca = MCA(ind_sup=list(range(27,races_canines.shape[0])),quanti_sup=7,quali_sup=6)
    > res_mca.fit(races_canines)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pd.read_excel(DATASETS_DIR/"races_canines.xlsx",header=0,index_col=0)
    data.name = "races_canines"
    return data

def load_tea():
    """
    tea
    ---

    Description
    -----------
    The data used here concern a questionnaire on tea. We asked to 300 individuals how they drink tea (18 questions), what are their product's perception (12 questions) and some personal details (4 questions).

    Usage
    -----
    > from scientisttools load_tea
    > tea = load_tea()

    Format
    ------
    A data frame with 300 rows and 36 columns. Rows represent the individuals, columns represent the different questions. The first 18 questions are active ones, the 19th is a supplementary quantitative variable (the age) and the last variables are supplementary categorical variables.

    Source
    ------
    The tea dataset from FactoMineR.

    Examples
    --------
    # Load tea dataset
    > from scientisttools import load_tea
    > tea = load_tea()
    >
    > from scientisttools import MCA
    > res_mca = MCA(quanti_sup=18, quali_sup=list(range(19,36)))
    > res_mca.fit(tea)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"tea.rda")["tea"]
    data.name = "tea"
    return data

def load_poison():
    """
    Poison
    ------

    Description
    -----------
    The data used here refer to a survey carried out on a sample of children of primary school who suffered from food poisoning. They were asked about their symptoms and about what they ate.

    Usage
    -----
    > from scientisttools import load_poison
    > poison = load_poison()

    Format
    ------
    A data frame with 55 rows and 15 columns.

    Source
    ------
    The poison dataset from FactoMineR

    Examples
    --------
    > # Load poison dataset
    > from scientisttools import load_poison
    > poison = load_poison()
    >
    > from scientisttools import MCA
    > res_mca = MCA(n_components=5,ind_sup=list(range(50,55)),quali_sup = [2,3],quanti_sup =[0,1])
    > res_mca.fit(poison)

    Author(s)
    --------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"poison.rda")["poison"]
    data.name = "poison"
    return data

def load_mushroom():
    """
    Mushroom
    --------

    Usage
    -----
    > from scientisttools import load_mushroom
    > mushroom = load_mushroom()

    Source
    ------
    The Mushroom uci dataset. See https://archive.ics.uci.edu/dataset/73/mushroom

    Examples
    --------
    > # Load mushroom dataset
    > from scientisttools import load_mushroom
    > mushroom = load_mushroom()
    >
    > from scientisttools import MCA
    > res_mca = MCA()
    > res_mca.fit(mushroom)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pd.read_excel(DATASETS_DIR/"mushroom.xlsx")
    data.name = "mushroom"
    return data

def load_music():
    """
    Music
    ----

    Description
    -----------
    The data concerns tastes for music of a set of 500 individuals. It contains 5 variables of likes for music genres (french pop, rap, rock, jazz and classical), 2 variables about music listening and 2 additional variables (gender and age).

    Usage
    -----
    > from scientisttools import load_music
    > music = load_music()

    Format
    ------
    A data frame with 500 observations and 7 variables

    Source
    ------
    The Music dataset in R GDAtools packages

    Examples
    --------
    > # Load music dataset
    > from scientisttools import load_music, SpecificMCA
    > music = load_music()
    >
    > excl = {"FrenchPop" : "NA", "Rap" : "NA" , "Rock" : "NA", "Jazz" : "NA","Classical" : "NA"}
    > res_spemca = SpecificMCA(n_components=5,excl=excl)
    > res_spemca.fit(music)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"Music.RData")["Music"]
    data.name = "Music"
    return data

def load_gironde(which="all"):
    """
    gironde
    -------

    Description
    -----------
    a dataset with 542 individuals and 27 columns

    Examples
    --------
    ```python
    >>> from scientisttools import load_gironde, PCAMIX
    >>> gironde = load_gironde()
    >>> res_pcamix = PCAMIX()
    >>> res_pcamix.fit(gironde)
    ```

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
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

def load_tennis():
    """
    Tennis
    ------

    Usage
    -----
    > from scientisttools import load_tennis
    > tennis = load_tennis()

    Examples
    --------
    > # Load tennis dataset
    > from scientisttools import load_tennis
    > tennis = load_tennis()
    >
    > from scientisttools import FAMD
    > res_famd =  FAMD(n_components=2,ind_sup=list(range(16,tennis.shape[0])),quanti_sup=7)
    > res_famd.fit(tennis)

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pd.read_excel(DATASETS_DIR/"tennisplayers.xlsx",index_col=0)
    return data

################################## Multiple Fcator Analysis (MFA) ########################################
def load_burgundy_wines():
    """
    Burgundy wines dataset
    ----------------------

    Source: https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf

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
        columns=pd.MultiIndex.from_tuples(
            [
                ("Expert 1", "Fruity"),
                ("Expert 1", "Woody"),
                ("Expert 1", "Coffee"),
                ("Expert 2", "Red fruit"),
                ("Expert 2", "Roasted"),
                ("Expert 2", "Vanillin"),
                ("Expert 2", "Woody"),
                ("Expert 3", "Fruity"),
                ("Expert 3", "Butter"),
                ("Expert 3", "Woody"),
            ],
            names=("expert", "aspect"),
        ),
        index=[f"Wine {i + 1}" for i in range(6)],
    )
    wines.insert(0, "Oak type", [1, 2, 2, 2, 1, 1])
    return wines

def load_wine():
    """
    Wine
    ----

    Description
    -----------
    The data used here refer to 21 wines of Val de Loire

    Usage
    -----
    > from scientisttools import load_wine
    > wine = load_wine()

    Format
    ------
    A data frame with 21 rows (the number of wines) and 31 columns: 
        - the first column corresponds to the label of origin, 
        - the second column corresponds to the soil, 
        - and the others correspond to sensory descriptors.
    
    Source
    ------
    The wine dataset from FactoMineR

    Examples
    --------
    > # Load wine data
    > from scientisttools import load_wine
    > wine = load_wine()
    > 
    > # Example of PCA
    > from scientisttools import PCA
    > res_pca = PCA(standardize=True,n_components=5,quanti_sup=[29,30],quali_sup=[0,1],parallelize=True)
    > res_pca.fit(wine)
    > 
    > # Example of MCA
    > from scientisttools import MCA
    > res_mca = MCA(quanti_sup = list(range(2,wine.shape[1])))
    > res_mca.fit(wine)
    > 
    > # Example of FAMD
    > from scientisttools import FAMD
    > res_famd = FAMD()
    > res_famd.fit(wine)
    > 
    > # Example of MFA
    > from scientisttools import MFA
    > res_mfa = MFA(n_components=5,group=group,group_type=["n"]+["s"]*5,var_weights_mfa=None,name_group = group_name,num_group_sup=[0,5],parallelize=True)
    > res_mfa.fit(wine)
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"wine.rda")["wine"]
    data.name = "wine"
    return data

def load_qtevie():
    """
    
    
    """
    data = pd.read_csv(DATASETS_DIR/"QteVie.csv",encoding="ISO-8859-1",header=0,sep=";",index_col=0)
    return data

########################################## MFACT
def load_mortality():
    """
    The cause of mortality in France in 1979 and 2006
    -------------------------------------------------

    Description
    -----------
    The cause of mortality in France in 1979 and 2006

    Usage
    -----
    > from scientisttools import load_mortality
    > mortality = load_mortality()

    Format
    ------
    A data frame with 62 rows (the different causes of death) and 18 columns. Each column corresponds to an age interval (15-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85-94, 95 and more) in a year. The 9 first columns correspond to data in 1979 and the 9 last columns to data in 2006. In each cell, the counts of deaths for a cause of death in an age interval (in a year) is given.

    Source
    ------
    The mortality dataset from FactoMineR

    Examples
    --------
    > # load mortality dataset
    > from scientisttools import load_mortality
    > mortality = load_mortality()
    >
    > from scientisttools import MFACT
    > res_mfact = MFACT()

    """
    data = pyreadr.read_r(DATASETS_DIR/"mortality.rda")["mortality"]
    data.name = "mortality"
    return data

def load_lifecyclesavings():
    """
    Intercountry Life-Cycle Savings Data
    ------------------------------------

    Description
    -----------
    Data on the savings ratio 1960 - 1970

    Usage
    -----
    > from scientisttools import load_lifecyclesavings
    > lifecyclesavings = load_lifecyclesavings()

    Format
    -----
    A data frame with 50 observations on 5 variables

    Source
    ------
    The LifeCycle Savings dataset from R datasets

    Examples
    --------
    > # Load lifecyclesavings dataset
    > from scientisttools import load_lifecyclesavings
    > lifecyclesavings = load_lifecyclesavings()
    >
    > from scientisttools import CCA
    > res_cca = CCA(lifecyclesavings,vars=[1,2])

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com    
    """
    data = pyreadr.read_r(DATASETS_DIR/"LifeCycleSavings.RData")["LifeCycleSavings"]
    data.name = "LifeCycleSavings"
    return data

# Sparse PCA datasets
def load_protein():
    """
    Protein
    -------

    Description
    -----------
    This dataset gives the amount of protein consumed for nine food groups in 25 European countries. The nine food groups are red meat (RedMeat), white meat (WhiteMeat), eggs (Eggs), milk (Milk), fish (Fish), cereal (Cereal), starch (Starch), nuts (Nuts), and fruits and vegetables (FruitVeg).

    Usage
    -----
    ```python
    >>> 
    ```
    > from scientisttools import load_protein
    > protein = load_protein()

    Format
    ------
    A numerical data matrix with 25 rows (the European countries) and 9 columns (the food groups)

    Source
    ------
    The protein dataset for sparsePCA R package

    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE duverierdjifack@gmail.com
    """
    data = pyreadr.read_r(DATASETS_DIR/"protein.RData")["protein"]
    data.name = "protein"
    return data

def load_vote():
    """
    Congressional Voting Records
    ----------------------------


    Usage
    -----
    > from scientisttools import load_vote
    > vote = load_vote()

    Source
    ------
    The Congressional Voting Records. See https://archive.ics.uci.edu/dataset/105/congressional+voting+records
    
    Examples
    --------
    > # Load vote dataset
    > from scientistools import load_vote
    > vote = load_vote()
    >
    > from scientisttools import CATVARHCA
    > X = vote.iloc[:,1:]
    > res_catvarhca =  CATVARHCA(n_clusters=2,diss_metric="cramer",metric="euclidean",method="ward",parallelize=True)
    > res_catvarhca.fit(X)
    
    Author(s)
    ---------
    Duvérier DJIFACK ZEBAZE djifacklab@gmail.com
    """
    data = pd.read_excel(DATASETS_DIR/"congressvotingrecords.xlsx")
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
    ```
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
    usarrests.index.name = None
    return usarrests