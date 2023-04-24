
import pandas as pd
import matplotlib.pyplot as plt


mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}
passings = [3, 7, 2]
passings_2 = [10, 11, 12]
calories = {"day1": 420, "day2": 380, "day3": 390}
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

myvar = pd.DataFrame(mydataset)
serie_panda = pd.Series(passings)
creat_etiqt = pd.Series(passings, index = ["x", "y", "z"])

# création d'un serie via un dictionnaire calorie
serie_via_dict = pd.Series(calories)

# afficher la première d'une Series
serie_via_dict[0]

# Creéation d'une serie via des éléménts d'un dictionnaire(day2 et day3)
create_series_spec = pd.Series(calories, index=["day2", "day3"])

# création d'une dataframes à partir de 2 series
creat_dataFrames_with2Series = pd.DataFrame(data)

# Pour renvoyer une ligne spécifique d'une dataframe on utilise <loc>
line_une_dataframes = creat_dataFrames_with2Series.loc[1]

# Pour renvoyer une ligne spécifique d'une dataframe on utilise <loc>
multiline_dataframes = creat_dataFrames_with2Series.loc[[0, 1]]

# Nommer les indexes d'une DataFrames
index_dataframes = pd.DataFrame(data, index=["day1", "day2", "day3"])

# Localiser un index nommé
indexNommes_dataframes = index_dataframes.loc["day1"]

# Charger des données depuis un fichier CSV dans le dataframe
charge_donne_csv_in_df = pd.read_csv('data.csv') 

# Charger des données depuis un fichier JSON dans le dataframe
charge_donne_json_in_df = pd.read_json('data.json') 

# Afficher l'intégralité d'une dataframe json ou csv
charge_donne_csv_in_df.to_string() 
charge_donne_json_in_df.to_string()

# modifier le nombre de ligne renvoyé dans l'affichage print()
pd.options.display.max_rows = 100

# -------------- Analyse de dataFrame ----------------------

# afficher un apperçu du dataframe en imprimant les 5 première ligne de l'entête
affiche_head_of_5lines =  charge_donne_csv_in_df.head(5)

# afficher un apperçu du dataframe en imprimant les 5 dernière ligne du dataframe
affiche_head_of_5LastLines =  charge_donne_csv_in_df.tail()

# afficher les info sur le dataframe
affiche_info_dataframe = charge_donne_json_in_df.info()

# -------------- Cleaning Data with cell Empty ----------------------

# Si une ligne contient une cellule vide pour nettoyer les données nous supprimons toute la ligne
read_csv_file_inDataframe = pd.read_csv('data.csv')

# suppresion de ligne contenant les cellule vide
cleanData_line_with_celEmpty = read_csv_file_inDataframe.dropna()   # creation d'un nouveau dataframe
#cleanData_line_with_celEmpty = read_csv_file_inDataframe.dropna(inplace=True)  # modifie le dataframe d'origine

# afficher integralité du dataframe via le to_string
all_dataframe_afterClean = cleanData_line_with_celEmpty.to_string()

# Pour modier le dataframe d'origine on utilise inplace=True dans le methode dropna()

# Pour inserer de nouvelle valeur dans toutes les cellules vides du dataframe on utilise la méthode fillna()
read_csv_file_inDataframe.fillna("trailhead4.wfokpckfroxp@example.com", inplace = True)

# pour inserer une valeur dans toutes les cellules vides d'une colonne spécifque du dataframe
read_csv_file_inDataframe_1 = pd.read_csv('data.csv')
read_csv_file_inDataframe_1["Username"].fillna("trailhead4.wfokpckfroxp@example.com", inplace = True)

# Remplacer dans une colonne de valeur numérique les cellules vides par une moyenne (mean()), median(), mode()
read_csv_file_inDataframe_2 = pd.read_csv('data.csv') 
# calcul de la valeur moyenne, mediane, et modale de la colonne QuotaAmount du dataframe 
#valeur_moyenne = read_csv_file_inDataframe_2["QuotaAmount"].mean() 
valeur_moyenne = read_csv_file_inDataframe_2["QuotaAmount"].median()
#valeur_moyenne = read_csv_file_inDataframe_2["QuotaAmount"].mode()[0]
# remplacement de la valeur moyenne dans les cellules vides de la colonne QuotaAmount du dataframe
read_csv_file_inDataframe_2["QuotaAmount"].fillna(valeur_moyenne, inplace = True)

#read_csv_file_inDataframe.fillna("trailhead4.wfokpckfroxp@example.com", inplace = True)
# --------------------------------------------------------------------

# -------------- Cleaning Data with wrong format ----------------------
# Certaines cellule de la colonne date presente le mauvais format convertissons les en Date via to_datetime()
read_csv_file_inDataframe_3 = pd.read_csv('data.csv')

# Convertion de la colonne StartDatede la dataframe en Date via to_datetime
read_csv_file_inDataframe_3['StartDate'] = pd.to_datetime(read_csv_file_inDataframe_3['StartDate'])

# supression de ligne avec les valeurs vide de la colonne StartDate du dataframe
read_csv_file_inDataframe_3.dropna(subset=['StartDate'], inplace = True)
# --------------------------------------------------------------------

# -------------- Cleaning Data with wrong data -----------------------
# pour corriger des données erronées on peut definir une valeur X au dela du quel toute valeur superieur est raplacé par Z
read_csv_file_inDataframe_4 = pd.read_csv('data.csv')

# pour corriger une valeur spécifique (indexe est celui de la valeur qu'on souhaite modifier dans la colonne specifique) 
#read_csv_file_inDataframe_4.loc[indexe, 'QuotaAmount'] = 150000

# Nous parcourons les indexes du dataframe 
for indexe in read_csv_file_inDataframe_4.index:
    # si un index de la colonne QuotaAmount du dataframme est superieur à 150000
    if read_csv_file_inDataframe_4.loc[indexe, 'QuotaAmount'] > 150000:
        # Nous remplacons la valeur sup 150000 par celle egale a 150000
        read_csv_file_inDataframe_4.loc[indexe, 'QuotaAmount'] = 150000

# suppression des données érronées  dans le dataframe
read_csv_file_inDataframe_5 = pd.read_csv('data.csv')
for indexe in read_csv_file_inDataframe_5.index:
    # si un index de la colonne QuotaAmount du dataframme est superieur à 150000
    if read_csv_file_inDataframe_5.loc[indexe, 'QuotaAmount'] > 150000:
        # nous supprimons toutes les lignes de la colonnes QuotaAmount
        read_csv_file_inDataframe_5.drop(indexe, inplace = True)
# --------------------------------------------------------------------

# -------------- Cleaning Data with Removing Duplicates -----------------------
read_csv_file_inDataframe_6 = pd.read_csv('data.csv')

# la methode duplicated() renvoie true ou false si chaque ligne est un doublons dans le dataframe
affichage_lineDuplicate = read_csv_file_inDataframe_6.duplicated()

# supression de doublon via drop_duplicates()
read_csv_file_inDataframe_6.drop_duplicates(inplace = True)
# -----------------------------------------------------------------------------

# ----------------------------  Data Correlations -----------------------------
# Determinons la correlation entre les colonnes de notre dataframe
read_csv_file_inDataframe_7 = pd.read_csv('data_2.csv')
# recherche de correlation via la methode corr()
affiche_correlation = read_csv_file_inDataframe_7.corr()
# -----------------------------------------------------------------------------

# ----------------------  pandas plotting(traçage) ---------------------------
# Determinons la correlation entre les colonnes de notre dataframe
read_csv_file_inDataframe_8 = pd.read_csv('data_2.csv')

# Affichage du diagramme via plot()
read_csv_file_inDataframe_8.plot()
plt.show()

# affichage pour un nuage de point avec le repère x = 'Duration', y = 'Calories'
#read_csv_file_inDataframe_8.plot(kind = 'scatter', x = 'Duration', y = 'Calories')
read_csv_file_inDataframe_8.plot(kind = 'scatter', x = 'Duration', y = 'Maxpulse')
plt.show()

# Affichage des histogramme 
read_csv_file_inDataframe_8['Duration'].plot(kind = 'hist')
plt.show()
# -----------------------------------------------------------------------------

print(serie_panda)
print("serie_panda[1]:", serie_panda[1])
print(creat_etiqt)
print("creat_etiqt[\"x\"]:", creat_etiqt["x"])
print(serie_via_dict)
print("create_series_spec \n", create_series_spec)
print("creat_dataFrames_with2Series \n", creat_dataFrames_with2Series)
print("line_une_dataframes \n", line_une_dataframes)
print("multiline_dataframes \n", multiline_dataframes)
print("index_dataframes \n", index_dataframes)
print("indexNommes_dataframes \n", indexNommes_dataframes)
print("charge_donne_csv_in_df \n", charge_donne_csv_in_df)
print("charge_donne_csv_in_df \n", charge_donne_csv_in_df.to_string())
print(pd.__version__)
print(pd.options.display.max_rows) 
print("affiche_head_of_5lines \n", affiche_head_of_5lines)
print("affiche_head_of_5LastLines \n", affiche_head_of_5LastLines)
print("affiche_info_dataframe \n", affiche_info_dataframe)

print("suppresion de ligne contenant les cellule vide \n", all_dataframe_afterClean)
print("remplacement dans dataframe de cellule vide par une valeur \n", read_csv_file_inDataframe.tail(10))
print("remplacement dans dataframe de cellule vide par une valeur \n", read_csv_file_inDataframe_1.tail(10))
print("remplacement dans dataframe de cellule vide par une valeur moyenne \n", read_csv_file_inDataframe_2.tail(10))
print("convertion de la colonne StartDate du DF en Date  \n", read_csv_file_inDataframe_3.to_string())
print("Remplacement de valeur erronée par une condition  \n", read_csv_file_inDataframe_4.to_string())
print("Suppression de valeur erronée par une condition  \n", read_csv_file_inDataframe_5.to_string())
print("Renvoie de doublon dans le dataframe \n", affichage_lineDuplicate)
print("Suppression de doublon dans le dataframe \n", read_csv_file_inDataframe_6.to_string())
print("Correlation entre les colonnes d'un dataframe \n", affiche_correlation)

#print("Affichage du diagramme du dataframe \n", affiche_tracage)



