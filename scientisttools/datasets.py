# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import pyreadr 



# https://husson.github.io/data.html
# https://r-stat-sc-donnees.github.io/liste_don.html

### Principal Components Analysis ############################

def load_decathlon():
    """The Decathlon dataset from FactoMineR."""
    data = pd.read_excel("decathlon.xlsx",header=0,index_col=0)
    data.name = "decathlon"
    return data

def load_decathlon2():
    """The Decathlon dataset from Factoextra."""
    data = pd.read_excel("decathlon2.xlsx",header=0,index_col=0)
    data.name = "decathlon2"
    return data

def load_autos():
    """Autos 2005 - Données sur 40 voitures"""
    data = pd.read_excel("autos2005.xls",header=0,index_col=0)
    data.name = "autos_2005"
    return data

def load_temperature():
    data = pd.read_excel("temperature.xlsx",header=0,index_col=0)
    data.name = "temperature"
    return data

def load_temperature2():
    data = pd.read_excel("temperature_acp.xlsx",header=0,index_col=0)
    data.name = "temperature2"
    return data

############################################# Correspondance Analysis ########################################""

def load_woman_work():
    """"""
    data = pd.read_csv("women_work.txt",sep="\t")
    data.name = "woman_work"
    return data

def load_femmes_travail():
    """"""
    data = pd.read_csv("femme_travail.csv",delimiter=";",encoding =  "cp1252",index_col =0)
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
    data = pyreadr.read_r("housetasks.rda")["housetasks"]
    data.name = "housetasks"
    return data

###################################### Multiple Correspondance Analysis #########################################

def load_races_canines():
    """"""
    data = pd.read_excel("races_canines.xls",header=0,index_col=0)
    data.name = "races_canines"
    return data

def load_races_canines2():
    """"""
    data = pd.read_excel("races_canines2.xlsx",header=0,index_col=0)
    data.name = "races_canines2"
    return data

def load_races_canines3():
    """
    
    """
    data = pd.read_excel("/races_canines_acm.xlsx",header=0,index_col=0)
    data.name = "races_canines3"
    return data

def load_tea():
    """"""
    data = pd.read_excel("tea.xlsx",header=0,index_col=0)
    data.name = "tea"
    return data

################################## Factor Analysis of Mixed Data #####################################"

def load_autos2():
    """FAMD Data - Données sur 45 voitures"""
    data = pd.read_excel("autos2005_afdm.xlsx",header=0,index_col=0)
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

def load_wine():
    """
    
    
    """
    # Load Data
    url = "http://factominer.free.fr/factomethods/datasets/wine.txt"
    wine = pd.read_table(url,sep="\t")

    # Transform
    wine2 = pd.DataFrame(data=wine.values,
                         columns = pd.MultiIndex.from_tuples(
                        [
                            ("others","Label"),
                            ("others","Soil"),
                            ("before shaking","Odor.Intensity"),
                            ("before shaking","Aroma.quality"),
                            ("before shaking","Fruity"),
                            ("before shaking","Flower"),
                            ("before shaking","Spice"),
                            ("vision","Visual.intensity"),
                            ("vision","Nuance"),
                            ("vision","Surface.feeling"),
                            ("after shaking","Odor.intensity"),
                            ("after shaking","Quality.of.odour"),
                            ("after shaking","Fruity"),
                            ("after shaking","Flower"),
                            ("after shaking","Spice"),
                            ("after shaking","Plante"),
                            ("after shaking","Phenolic"),
                            ("after shaking","Aroma.intensity"),
                            ("after shaking","Aroma.persistency"),
                            ("after shaking","Aroma.quality"),
                            ("gustation","Attack.intensity"),
                            ("gustation","Acidity"),
                            ("gustation","Astringency"),
                            ("gustation","Alcohol"),
                            ("gustation","Balance"),
                            ("gustation","Smooth"),
                            ("gustation","Bitterness"),
                            ("gustation","Intensity"),
                            ("gustation","Harmony"),
                            ("overall judgement","Overall.quality"),
                            ("overall judgement","Typical")
                        ]
        ))
    wine2.index= wine.index
    groups = wine2.columns.levels[0].drop(["others","overall judgement"]).tolist()
    for g in groups:
        wine2[g] = wine2[g].astype("float")
    
    return wine2

def load_qtevie():
    """
    
    """
    qtevie = pd.read_csv("QteVie.csv",encoding="ISO-8859-1",header=0,sep=";",index_col=0)
    qtevie.info()

    data = pd.DataFrame(qtevie.values,
                    columns=pd.MultiIndex.from_tuples(
                        [
                            ("weel_being","Logements sans sanitaires"),
                            ("weel_being","Coût logement"),
                            ("weel_being","Nb pièces par personne"),
                            ("weel_being","Revenu ménages"),
                            ("weel_being","Patrimoine financier"),
                            ("employment","Taux emploi"),
                            ("employment","Sécurité emploi"),
                            ("employment","Chômage longue durée"),
                            ("employment","Revenus moyens activité"),
                            ("employment","Horaires travail lourds"),
                            ("pleasure","Qualité réseau social"),
                            ("pleasure","Satisfaction sur la vie"),
                            ("pleasure","Temps aux loisirs et à soi"),
                            ("health_and_security","Pollution atmosphérique"),
                            ("health_and_security","Qualité eau"),
                            ("health_and_security","Espérance de vie"),
                            ("health_and_security","Auto-évaluation état de santé"),
                            ("health_and_security","Taux agression"),
                            ("health_and_security","Taux homocides"),
                            ("education","Niveau instruction"),
                            ("education","Compétences élèves"),
                            ("education","Années scolarité"),
                            ("region","Région")
                        ]
                    ))
    data.index = qtevie.index
    groups = data.columns.levels[0].drop(["region"]).tolist()
    for g in groups:
        data[g] = data[g].astype("float")
    return data

########################################## Autres datasets

def load_poison():
    data = pyreadr.read_r("poison.rda")["poison"]
    data.name = "poison"
    return data