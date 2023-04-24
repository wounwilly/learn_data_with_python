
import pandas as pd
import numpy as np
import sys
import os





# -------------------- Qu'est-ce que la science des données ? ----------------------
# (Data Scientist): aide les entreprise à prendre les decisions basées sur les données
# afin d'améliorer leurs activités.

# (La science des données): est une combinaison de plusieurs disciplines qui utilise les 
# statistiques, l'analyse de données et l'apprentissage automatique pour analyser les données
# et en extraire des connaissances et des idées.

# Science des données: consiste à collecter, analyser et prendre des décisions sur les données 
# Elle permet de trouver des modèles dans les données grâce à l'analyse et à faire des prévions futures.

# grâce à la data science les entreprises peuvent:
# prendre de meilleurs decisions
# Analyse prédictive
# Découvrir des modèle ou des information cachées dans les données
#-----------------------------------------------------------------------------------

# ------------------ Où la science des données est-elle nécessaire ? ----------------
# Utilisé dans de nombreuse industrie dans le monde telsque banque, la santé et la fabrication.
# Domaine où la science des données est necessaire:
# La planification d'itinéraire
# Prévision des retards de vol/navire/train
# creation des offres promotionnelles
# Prevoire les chiffre d'affaire d'une entreprise

# La science des données est applicable à toutes les parties d'un entreprise où les données sont disponibles.
#-------------------------------------------------------------------------------------

# ---------------------- Comment fonctionne un Data Scientist ? ----------------------
# un data scientist doit disposer d'un expertise dans plusieurs domaine:
# En apprentissage automatique 
# en statistique 
# mathématiqe
# base de données

# Son rôle est de trouver des modèle de donnée, pour cela il doit organiser les données dans un format standard.
#fonctionnent d'un data scientist:
# (Poser les bonne question) permet de comprendre la problématique du métier 
# (Explorer et collecter les données) à partir d'une base de données, des logs, commentaires
# (Extraire les données) en les transformant dans un format standard
# (Nettoyer les données) en supprimant les valeurs erronées
# (Rechercher et remplacer les valeurs manquantes) par exple par la valeur moyenne
# (Normaliser les données) les mettre à l'echelle d'une plage pratique
# (Analysez les données), trouvez des modèles et faites des prévisions futures
# (Représenter le résultat) de manière à ce que l'ntreprise comprenne
#-------------------------------------------------------------------------------------

# ------------------- Science des données - Qu'est-ce que les données ? --------------------
# (Les données) sont une collection d'information
# L'objectif de la data science est:
# de structurer les données, 
# les rendre interprétables 
# et facile à utiliser

# On distingue deux types de données:
## données structurées (sont arganisées et facile  à exploiter)
## données non structurées (ne sont pas organisées,nous devons les organiser à fin de les analyser)

### comment structurer les données ?
# La structuration des données peut se faire via un tableau en python
Array = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
print("Structuration des données en Python via un tableau:", Array)
#--------------------------------------------------------------------------------------------

# -------------------- Science des données - Table de base de données -----------------------
## tableau de base de données 
# Une table dans une base de données est une table des données structurées.

# La structure d'une table de base de données comprend:
## des lignes représente l'horizontale des données 
## des colonnes représente la verticale des données

## Les variables
# C'est quelque chose qui peut être mésuré ou compté exple les heures, des nombres
# chaque colonne d'une base de données represente une variable 
# et chaque ligne une varaition de la variable ou une observation (sachant que la première ligne constitue l'étiquette) 
# on parle d'étiquette ou nom de la variable.
#--------------------------------------------------------------------------------------------

# -------------------------------- Science des données et Python ----------------------------
# on utiliserons python en data science notament les bibliothèque tels que:
# (pandas) pour les données structurées: il permet la préparation des données, la creation des dataframe et importation des fichier
# (Numpy) biblio mathématique: qui possède un puissant tableau array à N dimension, fiaire algèbre lineaire, etc
# (Matplotlib) biblio pour visualiser des données 
# (Scipy) biblio qui contient des modules de l'algèbre linéaire
#--------------------------------------------------------------------------------------------

# --------------------------- Science des données - Python DataFrame ------------------------
## Dataframe avec pandas
# un bloc de données est une représentation structurée de données

donnee = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}

df = pd.DataFrame(data=donnee)                                          # creation du dataframe via DataFrame()
print("creation d'un dataframe ou bloc de donnée de 3 colonnes et 5 lignes avec pandas:", df)

# compter le nombre de colonne nous utiliserons shape[1]
# shape[1] parce que en python le nombre de colonne commence par 1
nombre_colonne_dataframe = df.shape[1]
print("compter le nombre de colonne:", nombre_colonne_dataframe)

# compter le nombre de lignes nous utiliserons shape[0]
# shape[0] parce que en python le nombre de ligne commence par 0.
nombre_ligne_dataframe = df.shape[0]
print("Compter le nombre de lignes:", nombre_ligne_dataframe)
#--------------------------------------------------------------------------------------------

# ----------------------------- Fonctions de science des données ----------------------------
# comme fonction nous utiliserons essentiellement 3 fonctions max(), min(), mean()

## La fonction max()
# la fonction en python permet d'utiliser la valeur la plus élevée du tableau

Average_pulse_max = max(80, 85, 90, 95, 100, 105, 110, 115, 120, 125)
print ("La valeurr max de la liste est:", Average_pulse_max)

## La fonction min()
# la fonction en python permet d'utiliser la valeur la plus base du tableau

Average_pulse_min  = min(80, 85, 90, 95, 100, 105, 110, 115, 120, 125)
print ("La valeur min de la liste est:", Average_pulse_min)

## La fonction mean()
# la fonction en python permet de déterminer la valeur moyenne d'un tableau

Calorie_burnage = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330]
Average_calorie_burnage = np.mean(Calorie_burnage)
print("La valeur moyenne du tableau:", Average_calorie_burnage)
#--------------------------------------------------------------------------------------------

# ----------------Fonctions de science des données - Préparation des données ---------------
# Avant d'analyser les données, elle doivent être extraite, rendu propre avent de l'utiliser

## Extraction et lecture des données avec Pandas
# Avant l'analyse des données ceci doivent être importées/extraite
# Via pandas nous importerons des données via le fichier csv

# importation des données depuis un fichier csv par pandas
# header=0 signifique que les nom des colonnes doivent se trouver à la première ligne ou ligne = 0
# sep="," le séparateur des valeurs dans le fichier csv est ici des virgules
# on peut utiliser la fonction head() qui n'affichera que par defaut les 5 premères lignes

csv_data = pd.read_csv("data.csv", header=0, sep=",")
six_premièreligne = csv_data.head(6)                    # afficher les 6 premières lignes du fichier ccsv
print("inportation des données du fichier data.csv: \n", six_premièreligne)

## Nettoyage des données
### Supprimer les lignes vides
# nous utiliserons la fonction dropna() 
# avec axis=0 pour supprimer les lignes avec les valeurs NaN

csv_data.dropna(axis=0, inplace=True)
print()
print("Suppresion des lignes avec des valeurs nulles dans le df: \n", csv_data)

## Catégorie de données
# pour analyser les données nous devons connaitre le type de données 
# (données numérique) contiennent des valeurs numériques exple 2.5
# (données catégoriel) contiennent des valeurs qui peuvent être mésurée par rapport à d'autre  exple couleur
# (donnée ordinal) contiennent des données catégorielles exple note scolaire A

## Types de données
# La fonction info()permet de determiner le type de données
types_de_donnee = csv_data.info()
print()
print("Le type de donnée:", types_de_donnee)

## convertir un objet en float64 nous utiliserons astype()
#csv_data["StartDate"] = csv_data['StartDate'].astype(float)
types_de_donnee1 = csv_data.info()
print()
print("Convertion d'un type objet en type float:", types_de_donnee1)

## Analyser les données
# Aprèss le nettoyage des données nous pouvons procéder à son analyse
# nous utiliserons la fonction (describes()) pour résumer les données 
analyse_donnee = csv_data.describe()
print()
print("Analyse des données d'un tableau avec la fonction describe: \n", analyse_donnee)

#--------------------------------------------------------------------------------------------