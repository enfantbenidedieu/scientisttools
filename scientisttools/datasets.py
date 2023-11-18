# -*- coding: utf-8 -*-
from __future__ import annotations
import pathlib
import pandas as pd
import pyreadr 

DATASETS_DIR = pathlib.Path(__file__).parent / "data"

# https://husson.github.io/data.html
# https://r-stat-sc-donnees.github.io/liste_don.html

### Principal Components Analysis ############################

def load_decathlon():
    """The Decathlon dataset from FactoMineR."""
    data = pd.read_excel(DATASETS_DIR/"decathlon.xlsx",header=0,index_col=0)
    data.name = "decathlon"
    return data

def load_decathlon2():
    """The Decathlon dataset from Factoextra."""
    data = pd.read_excel(DATASETS_DIR/"decathlon2.xlsx",header=0,index_col=0)
    data.name = "decathlon2"
    return data

def load_autos():
    """Autos 2005 - Données sur 40 voitures"""
    data = pd.read_excel(DATASETS_DIR/"autos2005.xls",header=0,index_col=0)
    data.name = "autos_2005"
    return data

def load_temperature():
    data = pd.read_excel(DATASETS_DIR/"temperature.xlsx",header=0,index_col=0)
    data.name = "temperature"
    return data

def load_temperature2():
    data = pd.read_excel(DATASETS_DIR/"temperature_acp.xlsx",header=0,index_col=0)
    data.name = "temperature2"
    return data

############################################# Correspondance Analysis ########################################""

def load_woman_work():
    """"""
    data = pd.read_csv(DATASETS_DIR/"women_work.txt",sep="\t")
    data.name = "woman_work"
    return data

def load_femmes_travail():
    """"""
    data = pd.read_csv(DATASETS_DIR/"femme_travail.csv",delimiter=";",encoding =  "cp1252",index_col =0)
    data.name = "femmes_travail"
    return data

def load_housetasks():
    """House tasks contingency table

    Description
    -----------
    A data frame containing the frequency of execution of 13 house tasks in the couple. This table is also available in ade4 package.

    Format
    ------
    A data frame with 13 observations (house tasks) on the following 4 columns.
    Wife : a numeric vector
    Alternating : a numeric vector
    Husband : a numeric vector
    Jointly : a numeric vector
    """
    data = pyreadr.read_r(DATASETS_DIR/"housetasks.rda")["housetasks"]
    data.name = "housetasks"
    return data

###################################### Multiple Correspondance Analysis #########################################

def load_races_canines():
    """"""
    data = pd.read_excel(DATASETS_DIR/"races_canines.xls",header=0,index_col=0)
    data.name = "races_canines"
    return data

def load_races_canines2():
    """"""
    data = pd.read_excel(DATASETS_DIR/"races_canines2.xlsx",header=0,index_col=0)
    data.name = "races_canines2"
    return data

def load_races_canines3():
    """"""
    data = pd.read_excel(DATASETS_DIR/"races_canines_acm.xlsx",header=0,index_col=0)
    data.name = "races_canines3"
    return data

def load_tea():
    """"""
    data = pd.read_excel(DATASETS_DIR/"tea.xlsx",header=0,index_col=0)
    data.name = "tea"
    return data

################################## Factor Analysis of Mixed Data #####################################"

def load_autos2():
    """FAMD Data - Données sur 45 voitures"""
    data = pd.read_excel(DATASETS_DIR/"autos2005_afdm.xlsx",header=0,index_col=0)
    data.name = "autos_2005"
    return data

################################## Multiple Fcator Analysis (MFA) ########################################

def load_burgundy_wines():
    """Burgundy wines dataset.

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

########################################## Autres datasets

def load_poison():
    data = pyreadr.read_r(DATASETS_DIR/"poison.rda")["poison"]
    data.name = "poison"
    return data