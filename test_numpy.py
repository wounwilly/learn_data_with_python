import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
from math import log


# Creation d'un tableau sous Numpy via la methode array() en passant une liste comme argument à array()
tab_numpy = np.array([1, 2, 3, 4, 5])

# tableau numpy via array() en passant un tuple comme argument à array
tab_numpy1 = np.array((1, 2, 3, 4, 5))

# creation d'un tableau 0-D avec la valeur 42
tab_0_D = np.array(42)

# creation d'un tableau 1-D(unidimensionnel) avec un tab 0-D ou une liste
tab_1_D =  np.array([10, 15, 20, 25, 30, 35, 40])

# creation d'un tableau 2D(bidimensionnel) avec deux tab 1-D ou deux listes
tab_2D =  np.array([[1, 2, 3, 4, 5], [9, 8, 7, 6, 1]])

# creation d'un tableau 3D(tridimensionnel) avec deux tab 2D ou deux listes
tab_3D =  np.array([[[1, 2, 3, 4, 5], [9, 8, 7, 6, 1]], [[11, 12, 13, 14, 15], [19, 18, 17, 16, 11]]])

# Pour verifier la dimension d'un tableau l'attribut ndim retourne un chiffre 
dim_tab = tab_3D.ndim

# creation d'un tableau de xD via l'attribut (ndmin = x)
tab_xD = np.array([1, 2, 3, 4], ndmin=4)

# ---------------------------- Accès aux elements d'un tableau Numpy ----------------------
# Accès au premier élémént d'un tableau 1D
first_elem = tab_1_D[0]

# Accès aux éléménts d'un tableau 2D
first_elem_2D = tab_2D[0]                                 # accès à la première liste du tab 2D
first_elem_2D_ofFisrtList = tab_2D[0, 1]                  # Accédez à l'élément de la 1ere ligne, 2ème colonne
secondLine_of_5colum = tab_2D[1, 4]                       # Accédez à l'élément de la 2e ligne, 5e colonne

# Accéder au elements d'un tableau 3D
thirdElement_of_2ndTab_of_firstTab = tab_3D[0, 1, 2]      # Accédez au troisième élément du deuxième tableau du premier tableau 
#------------------------------------------------------------------------------------------

# ---------------------------- Tranchage de tableau NumPy ---------------------------------
# Découpez les éléments d'un tab_1D de l'éléments de l'index 1 à l'index 5 du tableau suivant
decoup_1par5 = tab_1_D[1:5]

# Découpez de l'index 2 à la fin du tableau
decoup_2parfin = tab_1_D[2:]

# Découpez du début à l'index 4
decoup_debutpar4 = tab_1_D[:4]

# Découpez negatif du début à l'index -3 à -1
decoupNegtf_neg3Parneg1 = tab_1_D[-3:-1]

# Renvoie tous les éléments de l'index 1 à l'index 4  en utilisant une marche de 2
marche_1Par4par2 = tab_1_D[1:4:2]

# Renvoie tous les éléments en utilisant une marche de 2
marche_Parpar2 = tab_1_D[::2]

# Découpage du tableau 2D
# tab2D decoupez le 2ème élément, découpez de l'index 1 à l'index 4
decoup2D_1par4 = tab_2D[1, 1:4]

# tab2D decoupez les deux éléments, retournez l'index 2
decoup2D_2Elmtparindex3 = tab_2D[0:2, 3]

# tab2D decoupez les deux éléments, découpez de l'index 1 à l'index 3
decoup2D_2Elmt1par3 = tab_2D[0:2, 1:3]
#------------------------------------------------------------------------------------------

# ---------------------------- Types de données NumPy ---------------------------------
# Verifier le type d'un array
dtype_array = tab_1_D.dtype

# vérification du type d'une liste de chaine
list_chaine = np.array(['apple', 'banana', 'cherry'])
type_listChaine = list_chaine.dtype

# Création d'un tableau de type string avec dtype 
tab_chaine = np.array([1, 2, 3, 4], dtype='S')
type_tabChaine = tab_chaine.dtype

# Création d'un tableau d'entier de 4 Octects
tab_int_4octect = np.array([1, 2, 3, 4], dtype='i4')
type_tabInt4oct = tab_int_4octect.dtype

# Pour changer le type de donnée d'un tableau existant on utilise la methode astype() qui crée une copie du tab 
tab_of_floatt = np.array([1.2, 2.3, 3.4, 4.2])
new_tab_floatt_toInt = tab_of_floatt.astype('i')        # ou astype(int)
type_tabfloaot_toInt = new_tab_floatt_toInt.dtype

# Convertion ou changer le type de donnée de float à entier
tab_float = np.array([1.1, 2.1, 3.1])
tab_int = tab_float.astype(int)
type_tabInt = tab_int.dtype

# Convertion ou changer le type de donnée des entiers à bool 
tab_entier = np.array([1, 0, 3])
tab_bool = tab_entier.astype(bool)
#------------------------------------------------------------------------------------------

# ----------------------------------- Array Copy vs View ---------------------------------
# Creation d'un tableau d'entier dans Numpy
arr = np.array([1, 2, 3, 4, 5])

# Copy de ce tableau d'entier et modification du tableau d'origine via son premier index
x = arr.copy()                  # # modification apportée au tab d'origine n'affecte pas la copie
arr[0] = 42

# Creation d'une vue à partir du tableau d'origine puis modification du tableau d'origine
y = arr.view()                  # modification apportée au tab d'origine affecte la vue
arr[0] = 50

# Pour verifier si une copie ou une vue possède ses données on utilise l'attribut base
have_dataCopy = x.base
have_dataView = y.base
#------------------------------------------------------------------------------------------

# ----------------------------------- Forme de tableau NumPy -------------------------------
# C'est le nombre d'élément de chaque dimension on le détermine via l'attribut shape
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
forme_tap2D = arr.shape

# creation d'un tab de dimension 5 avec ndmin verifions que la derniere à 4 valeurs
arr = np.array([1, 2, 3, 4], ndmin=5)
forme_tap2D_ndmin = arr.shape
#------------------------------------------------------------------------------------------

# ----------------------------- Remodelage du tableau Numpy -------------------------------
# C'est changer la forme d'un tableau le faire passer d'un 1D à un 2D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Convertir un tableau 1D en tab 2D avec 4 tableau de 3 elements
redim_tap1D_toTap2D = arr.reshape(4, 3)

# Convertir un tableau 1D en tab 3D avec 2 tableaux contenant 3 tableaux de 2 elements
redim_tap1D_toTap3D = arr.reshape(2, 3, 2)

# Verifions si le tableau renvoyé possède ses propres elements
have_dataRedim_tap1D_toTap2D = redim_tap1D_toTap2D.base         # il s'agit d'une vue(View)

# convertir un tableau de dimension 1D en tableau de dimension inconnu en passant la valeur -1
redim_tap1D_toTapinconnu = arr.reshape(2, 3, -1)

# Pour applatir un tableau en 1D on utilise reshape(-1)
arr = np.array([[1, 2, 3], [4, 5, 6]])

# convertir ou applatir un tab 2D en un tab 1D
applat_tab2D_toTab1D = arr.reshape(-1)
#------------------------------------------------------------------------------------------

# ----------------------------- Itérer des tableaux Numpy -------------------------------
# Itérer sur des tableau 1D
arr = np.array([1, 2, 3])
for x in arr:
  print(x)

# Itérer sur des tableau 2D
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
  for y in x:
    print(y)

# # Itérer sur des tableau 3D
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
  for y in x:
    for z in y:
      print(z)

# Pour parcourir n'importe quel tableau de dimension n on utilise la fonction nditer() et prend un tableau en argument
tab_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
for x in np.nditer(tab_arr):
  print(x)

# Itération énumérée avec ndenumerate() avec 1D permet de mentionner les séquences 1 par 1
arr = np.array([1, 2, 3])
for idx, x in np.ndenumerate(arr):
  print(idx, x)

# Itération énumérée avec ndenumerate() avec 2D
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for idx, x in np.ndenumerate(arr):
  print(idx, x)

# Itération sur un tableau avec différent type de données via nditer() et les arguments 
arr = np.array([1, 2, 3])
for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):           # op_dtypes permet de modifier le type de données du tableau
  print(x)
print(type(x))

# itération avec différente taille de pas on utilise toujours la fonction nditer() avec tab en arg
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for x in np.nditer(arr[:, ::2]):
  print(x)
#------------------------------------------------------------------------------------------

# ----------------------------- Jonction de tableaux Numpy -------------------------------
# c'est une action qui consiste à mettre le contenu de 2 ou +sieurs tan en un seul, elle se fait par des axes
# Pour le faire nous utiliserons la fonction concatenate()

# Jointure de deux tab 1D en seule tab
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))
print("jointure de 2 tab 1D en une tab 1D:", arr)       # [1 2 3 4 5 6]

# Jointure de deux tab 2D se fait via un axe ou la colonne
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)
print("Jointure de 2 tab 2D via axe=1:", arr)       # [[1 2 5 6][3 4 7 8]]

# Jointure par empilement via la methode stack() ici on empile colonne sur colonne
# sur des tab 1D
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1)
print("Jointure par empilement de 2 tab 1D avec stack() et axe=1:", arr)    # [[1 4][2 5][3 6]]

# Fonction d'assistance pour empiler le long des lignes on utilise hstack()
# Pour 2 tab 1D
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
# empilage de ligne avec la fonction d'assistance hstack sur des tab 1D 
arr = np.hstack((arr1, arr2))
print("Avec hstack deux tab 1D produit un tab 1D:", arr)                      # [1 2 3 4 5 6]

# Fonction d'assisatnce pour empiler le long des colonnes via la fonction vstack()
# Pour 2 tab 1D produit un tab 2D
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
# empilage de colonne avec la fonction d'assistance vstack sur des tab 1D produit un tab 2D
arr = np.vstack((arr1, arr2))
print("Avec vstack deux tab 1D produit un tab 2D:", arr)                      # [[1 2 3] [4 5 6]]

# Fonction d'assisatnce pour empiler le long de la hauteur via la fonction dstack()
# Pour 2 tab 1D produit un tab 3D
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
# empilage de hauteur avec la fonction d'assistance dstack sur des tab 1D produit un tab 3D
arr = np.dstack((arr1, arr2))
print("Avec dstack deux tab 1D produit un tab 3D:", arr)                      # [[[1 4][2 5][3 6]]]

#------------------------------------------------------------------------------------------

# ---------------------------- Fractionnement de tableaux Numpy ---------------------------
# C'est une action qui consiste à fractionner des tableaux en plusieurs tableaux via array_split() 
# array_split prend comme argument un 'tab' et le 'nombre de tab en sortie' 

# Fractionnons un tab 1D en 3 tab 1D
arr = np.array([1, 2, 3, 4, 5, 6])
# Fractionnement d'un tab 1D via array_split en 3 tab 1D dans une liste
newarr = np.array_split(arr, 3)
print("Fractionnement d'un tab 1D via array_split en 3 tab 1D dans une liste:", newarr)        #[array([1, 2]), array([3, 4]), array([5, 6])]
# Accéder à different tableau
for i in range(0, len(newarr)):
  print(f"tableau {i}:", newarr[i], "\n")

# Fractionnons un tab 2D en 3 tab 2D
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)
print("Fractionnement d'un tab 2D renvoie 3 tab 2D:", newarr)           # [array([[1, 2],[3, 4]]), array([[5, 6],[7, 8]]), array([[ 9, 10],[11, 12]])]

# on peut spécifier l'axe ou la ligne sur lequel réaliser la scission
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
# on obtient le meme resulat en utilisant hsplit
newarr1 = np.hsplit(arr, 3)
print("Fractionnement d'1 tab 2D en 3 tab 2D en spécifiant l'axe de scission:", newarr)
print("Fractionnement d'1 tab 2D en 3 tab 2D via hsplit:", newarr1)
#------------------------------------------------------------------------------------------

# ---------------------------- Recherche dans un tableaux Numpy ---------------------------
#La méthode where permet de faire des recherche dans une tableau sur un element specifique 
# et de renvoyer son idexe dans une liste  

# Recherche de la valeur 4 dans le tableau
arr = np.array([1, 2, 3, 4, 5, 4, 4])
# renvoie dans une liste tous indexes de 4 dans le tableau
x = np.where(arr == 4)
# trouver les index où les valeurs sont paire via le modulo 2
y = np.where(arr % 2 == 0)
# trouver les index où les valeurs sont impaires via le modulo 2
z = np.where(arr % 2 == 1)
print("Recherche de la valeur 4 dans le tableau", x)           # (array([3, 5, 6]),)
print("Recherche des valeurs paire dans le tableau", y)        # (array([1, 3, 5, 7]),)
print("Recherche des valeurs impaire dans le tableau", y)      # (array([0, 2, 4, 6]),)

# Recherche trié avec la méthode searchsorted() qui verifie dans un tableau où la valeur souhaitée
# peut être insérée et retourne l'index de son emplacement dans le tab
arr = np.array([6, 7, 8, 9, 11, 13])
indexe_gauche = np.searchsorted(arr, 7)
# on peut spécifier l'index le plus à droite avec side='right' par defaut c'est le coté gauche
indexe_droit = np.searchsorted(arr, 7, side='right')
# Recherche trie de plusieurs valeur se fait via une liste et renvoie une liste indexe
index_multiple = np.searchsorted(arr, [5, 10, 12])
print("La valeur x peut être insérée coté gauche à l'index:", indexe_gauche)
print("La valeur x peut être insérée coté droit à l'index:", indexe_droit)
print("Recherche trié de plusieurs valeur renvoie une liste:", index_multiple)
#------------------------------------------------------------------------------------------

# ------------------------- Trie de sortie dans un tableaux Numpy -------------------------
# C'est une méthode qui consiste à mettre les éléments d'un tableau dans une sequence ordonnée
# La fonction sort() nous permet de trier dans une certaine sequence ordonnée

# Pour un tab 1D
import numpy as np
arr = np.array([3, 2, 0, 1])
arr1 = np.array(['banana', 'cherry', 'apple'])
ordre_asc = np.sort(arr)
ordre_asc_tabString = np.sort(arr1)
print("Ranger les elements d'un tableau dans l'ordre croissant:", ordre_asc)
print("Ranger les elements d'un tableau dans l'ordre alphabetique:", ordre_asc_tabString)

# pour un tab 2D
arr = np.array([[3, 2, 4], [5, 0, 1]])
ordre_asc_tab2D = np.sort(arr)
print("Ranger les elements d'un tableau 2D dans l'ordre croissant:", ordre_asc_tab2D)       # [[2 3 4][0 1 5]]
#------------------------------------------------------------------------------------------

# ----------------------------- Filtre dans un tableaux Numpy -----------------------------
# Le filtering consiste à créer un nouveau tableau à partir de l'extrait 
# de certains elements d'un tableau existant

arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print("filtering:", newarr)

# Autre exemple
# Creation du tab de filtrage
filter_arr = []

# go through each element in arr
for element in arr:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
# tableau filtré
newarr = arr[filter_arr]
print("creation de filtre:", filter_arr)
print("Tableau filtré issue de d'un tableau existant", newarr)

# Création d'un filtre directement à partir d'un tableau
# Créez un tableau de filtres qui renverra uniquement les éléments pairs
arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]

print("creation de filtre:", filter_arr)
print("Création d'un filtre directement à partir d'un tableau", newarr)
#------------------------------------------------------------------------------------------

# Voir la version de numpy sous python
numpy_version = np.__version__

# -------------------------- Les nombres aléatoire dans Numpy -----------------------------
# un nombre aléatoire c'est un nombre qui ne peut être prédit logiquement.
# un nombre pseudo-aleatoire est nombre aléatoire généré par un programme informatique qui peut être prédit

# Pour générer des nombres aleatoire nous utiliserons numpy random
nbre_aleatoire =  random.randint(100)
print("Mon nombre aléatoire: ", nbre_aleatoire)

# Générer un flottant aléatoire entre 0 et 1
nbre_aleatoire_flottant = random.rand()
print("Mon nombre aléatoire flottant: ", nbre_aleatoire_flottant)

# Générer un tableau 1D aléatoire de (0 à 100) via randint() qui prend en argument size=(x)
tab1D_aleatoire = random.randint(100, size=(5))
print("Génération tab1D aléatoire: ", tab1D_aleatoire)

# Générer un tableau 2D aléatoire de (0 à 100) via randint() qui prend en argument size=(x, y) 
# Avec x:nbre tab1D dans le tab2D et y: nbre éléménts dans le tab1D
tab2D_aleatoire_3tab1D_5Elmts = random.randint(100, size=(3, 5))
print("Génération tab2D aléatoire de 3 tab1D contenant chacun 5 éléments: ", tab2D_aleatoire_3tab1D_5Elmts)

# Génération d'un tab1D contenant 5 elements flottants
tab1D_aleatoire_5float = random.rand(5)
print("Génération tab1D contenant 5 éléments flottant: ", tab1D_aleatoire_5float)

# Générez un tableau 2D avec 3 lignes, chaque ligne contenant 5 nombres flotant aléatoires de 0 à 1:
tab2D_3line_5elementsAleatoires = random.rand(3, 5)
print("Génération tab2D contenant 3 tab1D et 5elements aleatoire: ", tab1D_aleatoire_5float)

# Génération d'un nombre aléatoire basé sur un tableau via la methode choice()
nbreAleatoire_base_tableau = random.choice([3, 5, 7, 9, 12, 15])
print("Générer un nombre aléatoire basé sur un tableau:", nbreAleatoire_base_tableau)

# Générer un tableau 2D basé sur un tableau de nbre aleatoire via la methode choice()
# tab2D comprend 3 tab1D et 5 elements par tab1D
tab2D_base_tableauAleatoire = random.choice([3, 5, 7, 9, 12, 15], size=(3, 5))
print("Générer un tab2D basé sur un tableau de nbre aleatoire:", tab2D_base_tableauAleatoire)
#------------------------------------------------------------------------------------------

# -------------------------- Distribution aléatoire des données ---------------------------
# distribution des données est une liste de toutes les valeurs possibles 
# et de la fréquence à la quelle chaque valeur se produit
# une distribution aléatoire est un ensemble de nombre aleatoire qui suivent une certaine fonction de densité de probabilité
# fonction de densité de probabilité est une fonction qui décrit une proba continue cad la proba de toutes les valeurs d'un tableau


# Générez un tableau 1D contenant 100 valeurs, où chaque valeur doit être 3, 5, 7 ou 9 avec les proba de generation de chaque nbre du tableau
# La somme de tous les nombres générés doit être égale à 1
tab1D_de100Vlrs_avecprobaDeGeneration = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))
# tab2D de 3 tab1D et de 5elements par tab1D
tab2D_de100Vlrs_avecprobaDeGeneration = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))
print("Tab1D_100valeur_probaDeGeneration:", tab1D_de100Vlrs_avecprobaDeGeneration)
print("Tab2D_100valeur_probaDeGeneration:", tab2D_de100Vlrs_avecprobaDeGeneration)
#------------------------------------------------------------------------------------------

# ----------------------------------- Permutation aléatoire -------------------------------
# Une permutation fait référence à un arrangement d'élements. Dans Numpy nous utiliserons
# shuffle()et permutation(). Mélanger signifie changer la disposition des éléments sur place

# Mélangez aléatoirement les éléments du tableau
acreation_tab1D = np.array([1, 2, 3, 4, 5])
# Mélange aléatoire d'un tab1D
random.shuffle(acreation_tab1D)                     # Shuffle apporte des modifications au tableau d'origine
print("Création d'un tableau et melangeant de ses elements:", acreation_tab1D)

# Génère une permutation aléatoire des éléments du tableau
permutation_tab = random.permutation(acreation_tab1D)     # ne modifie pas le tab d'origine 
print("Création d'un tableau et permutation de ses elements:", acreation_tab1D)
#------------------------------------------------------------------------------------------

# ---------------------- Visualisation de distribution avec seaborn -----------------------
# seaborn est une bibliothèque matplotlib utilisée pour réaliser les visualisation d'une 
# distribution aléatoire

# exemple de tracer d'un distplot
sns.distplot([0, 1, 2, 3, 4, 5])
plt.show()

# Tracer un Distplot sans l'histogramme avec l'argument hist à false 
sns.distplot([0, 1, 2, 3, 4, 5], hist=False)
plt.show()
#------------------------------------------------------------------------------------------

# --------------------------- distribution normal ou gaussienne ---------------------------
# Encore appelle distribution de gausienne, la distribution normal est l'une des plus importante
# Elle correspond à la probabilité de nbreux évènements exple: rythme cardiaque, le score de qi
# nous utiliserons la méthode random.normal() elle prend 3 param 
# loc- (Mean) où le pic de la cloche existe
# scale- (Écart type) à quel point la distribution du graphique doit être plate
# size- La forme du tableau retourné

# Générez une distribution normale aléatoire de taille 2x3
dist_aleatoire_normal = random.normal(size=(2, 3))
print("Génération d'une distribution normale aléatoire", dist_aleatoire_normal)

# Générez une distribution normale aléatoire de taille 2x3 avec une moyenne à 1 et un écart type de 2
dist_aleatoire_normal_loc1_scale2 = random.normal(loc=1, scale=2, size=(2, 3))
print("Génération d'une distribution normale aléatoire avec moy =1 ecart type = 1:", dist_aleatoire_normal_loc1_scale2)

# Visualisation de la distribution normale
sns.distplot(random.normal(size=1000), hist=False)
plt.show()

# NB: la courbe d'une distribution normal est connu sous le nom de courbe en cloche.
#------------------------------------------------------------------------------------------

# --------------------------- distribution normal ou gaussienne ---------------------------
# Une distribution binomiale est une distribution discrète, qui decrit un scénario binaire.
# Nous utiliserons la methode binomial() qui prend 3 paramètres en entrés:
# n- nombre d'essais
# p- probabilité d'occurrence de chaque essai 
# size- La forme du tableau retourné

# Compte tenu de 10 essais pour le tirage au sort génèrent 10 points de données
creation_distrib_binomial = random.binomial(n=10, p=0.5, size=10)
print("Distribution binomiale: ", creation_distrib_binomial)

# Visualisation de la distribution binomiale
sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)
plt.show()

# La différence entre une distri normale et une binomial est que :
# la dist.Nnormal est continue
# la dist.binomial est discrète 
# NB: avec suffisament de points les deux distributions son similaire.

sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')

plt.show()
#------------------------------------------------------------------------------------------

# ----------------------------------- La loi de Poisson -----------------------------------
# La distribution de poisson est une distribution discrete, elle estime combien de fois un 
# événement peut se produire dans un temps spécifié. 2 paramètres sont pris en compte:
# lam- taux ou nombre connu d'occurrences
# size- La forme du tableau retourné

# Générez une distribution aléatoire 1x10 pour l'occurrence 2 
dist_aleatoire_1fois10_occurence2 = random.poisson(lam=2, size=10)
print("Générez une distribution aléatoire 1x10 pour l'occurrence 2:", dist_aleatoire_1fois10_occurence2)

# Visualisation de la distribution de Poisson
sns.distplot(random.poisson(lam=2, size=1000), kde=False)
plt.show()

# difference entre la distribution normale et celle de poisson
sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')

plt.show()

# difference entre la distribution binomial et celle de poisson
# distribution binomiale ne peut avoir que deux resultats possibles  n * p
# distribution de poisson peux avoir des résultats possibles et illimités. lam

sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')

plt.show()

# NB: n * p est proche de lam alors de resultat de l'affichage de la dist.poisson est presque égale à binomial
#------------------------------------------------------------------------------------------

# ---------------------------- La distribution normale -----------------------------------
# Permet de décrire la probabilité où chaque évèment à des chances égale de se produire, prend 3 paramètres 
# a- limite inférieure - par défaut 0 .0
# b- limite supérieure - valeur par défaut 1.0
# size- La forme du tableau retourné

# Création d'un échantillon de distribution uniforme 2x3
creation_dist_uniforme = random.uniform(size=(2, 3))
print(creation_dist_uniforme)

# Visualisation de la distribution uniforme
sns.distplot(random.uniform(size=1000), hist=False)
plt.show()
#------------------------------------------------------------------------------------------

# ---------------------------- La distribution logistique -----------------------------------
# La distribution logistique est utilisée pour décrire la croissance très utilisé dans les réseaux de neurones
# il prend trois paramètres 
# loc- signifie, où le pic est. 0 par défaut
# scale- l'écart type, la planéité de la distribution. Par défaut 1
# size- La forme du tableau retourné

# Dessinez des échantillons 2x3 à partir d'une distribution logistique avec une moyenne à 1 et une écart type de 2,0
creation_dist_logistique = random.logistic(loc=1, scale=2, size=(2, 3))
print("Distribution logistique moy=1 ecart type= 2:", creation_dist_logistique)

# Visualisation de la distribution logistique
sns.distplot(random.logistic(size=1000), hist=False)
plt.show()

# Différence entre distribution logique  et normale
sns.distplot(random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(random.logistic(size=1000), hist=False, label='logistic')

plt.show()

# les deux sont presque identique 
# La distribution logistique représente plus de possibilité d'occurrence d'un événement plus éloigné de la moyenne
# Pour une valeur d'échelle plus élevée (écart type), les distributions normale et logistique 
# sont presque identiques à l'exception du pic
#------------------------------------------------------------------------------------------

# ---------------------------- La distribution multinomiale -----------------------------------
# La distribution multinomiale est une généralisation de la distribution binomiale
# Il décrit les résultats de scénarios multinomiaux contrairement au binôme 
# où les scénarios ne doivent être que l'un des deux. Prend 3 paramètres en entrée:
# n- nombre de résultats possibles
# pvals- liste des probabilités de résultats
# size- La forme du tableau retourné

# Dessinez un échantillon pour lancer de dés 
dist_multinomiale = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print("Distribution multinomiale pour un dés:", dist_multinomiale)

# NB: les échantillons multinomiaux ne produiront PAS une seule valeur !
#  Ils produiront une valeur pour chacun pval.
#------------------------------------------------------------------------------------------

# ---------------------------- La distribution exponentielle -----------------------------------
# La distribution exponentielle est utilisée pour décrire le temps jusqu'au prochain événement
# Il prend deux paramètre en compte:
# scale- inverse du taux par défaut à 1,0
# size- La forme du tableau retourné

# Création d'une distribution exponentielle avec une échelle 2.0 avec une taille 2x3
Creation_dist_exponentiel = random.exponential(scale=2, size=(2, 3))
print("Création d'une distribution exponentielle avec une échelle 2.0", Creation_dist_exponentiel)

# Visualisation de la distribution exponentielle
sns.distplot(random.exponential(size=1000), hist=False)
plt.show()

# La diffirence entre la distribution de Poisson et exponentiel
# La distribution de Poisson traite du nombre d'occurrences d'un événement dans une période de temps
# la distribution exponentielle traite du temps entre ces événements
#------------------------------------------------------------------------------------------

# ------------------------- La distribution du carré du chi -------------------------------
# La distribution du chi carré est utilisée comme base pour vérifier l'hypothèse
# prend 2 paramètres:
# df- (degré de liberté)
# size- La forme du tableau retourné

# distribution du chi carré avec le degré de liberté 2 avec la taille 2x3 
dist_du_chi = random.chisquare(df=2, size=(2, 3))
print("Distribution du chi carré avec le degré de liberté 2", dist_du_chi)

# Visualisation de la distribution du chi carré
sns.distplot(random.chisquare(df=1, size=1000), hist=False)
plt.show()
#------------------------------------------------------------------------------------------

# --------------------------- La distribution de Rayleigh ---------------------------------
# La distribution de Rayleigh est utilisée dans le traitement du signal
#Il a deux paramètres :
# scale- (écart type) décide de l'aplatissement de la distribution par défaut 1.0).
# size- La forme du tableau retourné.

# distribution de Rayleigh avec une échelle de 2 avec une taille de 2x3 
dist_de_rayleigh = random.rayleigh(scale=2, size=(2, 3))
print("distribution de Rayleigh avec une échelle de 2", dist_de_rayleigh)

# Visualisation de la distribution de Rayleigh
sns.distplot(random.rayleigh(size=1000), hist=False)
plt.show()

# Similitude entre la distribution de Rayleigh et celle du chi carré
#------------------------------------------------------------------------------------------

# --------------------------- La distribution de Rayleigh ---------------------------------
# distribution de la la loi de Pareto, c'est-à-dire une distribution 80-20
# (20 % de facteurs entraînent 80 % de résultats). Il a deux paramètres :
# a- paramètre de forme.
# size- La forme du tableau retourné

#la distribution de Pareto avec la forme de 2 avec la taille 2x3 
dist_loi_pareto = random.pareto(a=2, size=(2, 3))
print("Distribution de Pareto avec la forme de 2", dist_loi_pareto)

# Visualisation de la distribution de Pareto
sns.distplot(random.pareto(a=2, size=1000), kde=False)
plt.show()
#------------------------------------------------------------------------------------------

# --------------------------- La distribution de Zipf -------------------------------------
# Les distributions Zipf sont utilisées pour échantillonner des données basées sur la loi de zipf
# Il a deux paramètres :
# a- paramètre de distribution.
# size- La forme du tableau retourné.

# distribution zipf avec le paramètre de distribution 2 avec une taille de 2x3
dist_zpif = random.zipf(a=2, size=(2, 3))
print("distribution zipf avec le paramètre de distribution 2:", dist_zpif)

# Visualisation de Zipf Distribution
x = random.zipf(a=2, size=1000)
sns.distplot(x[x<10], kde=False)
plt.show()

# NB: Échantillonnez 1000 points mais tracez seulement ceux avec une valeur < 10 
# pour un graphique plus significatif
#------------------------------------------------------------------------------------------

# --------------------------- Les fonctions universelles ufunc -------------------------------------
# sont des fonction NumPy qui opère sur les ndarray object, elle permette d'implémenter la vectorisation qui est
# plur rapide que l'itération sur les éléments. elles prennent des argument comme:
# (where) tableau booleen ou condition definissant où les operation doivent avoir lieu
# (dtype) qui est le type de retour des éléments
# (out)qui est tableau de sortie 

# La vectorisation
# c'est la convertion d'un intruction itérative en une operation vectorielle 

# Additionner deux listes avec la methode vectorielle via la methode add()
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
addition_liste = np.add(x, y)
print("Addition de 2 listes via la méthode vectorielle add() sur X et Y:", addition_liste)

# Creation de ses propre ufunc
# elle se fait avec le méthode frompyfunc() qui prend 3 arguments:
# (function) le nom de la fonction
# (inputs) le nombre d'arguments d'entrée (tableaux)
# (outputs) le nombre de tableaux de sortie

# Créez votre propre ufunc d'ajout 
def myadd(x, y):
    return x+y

# def de notre propre ufunc via frompyfunc
myUfunc_addition = np.frompyfunc(myadd, 2, 1)
add_viaUfunc = myUfunc_addition([1, 2, 3, 4], [5, 6, 7, 8])
print("Addition via la creation de ma propre ufunc:", add_viaUfunc)

# verifier s'il s'agit d'une ufunc
type_fonctionAdd = np.add
type_fonctionConcatenate= np.concatenate
print("Vérification du type d'une fonction:", type_fonctionAdd, type_fonctionConcatenate)

# instruction pour verifier si on a affaire à une (ufunc) ou pas 
if type(np.add) == np.ufunc:
  print("Il s'agit d'un ufunc")
else:
  print("Il ne s'agit pas d'une ufunc")
#------------------------------------------------------------------------------------------

# ------------------------- Arithmétique simple dans les ufunc ----------------------------
# arithmétique conditionnelle permet de définir les conditions dans lequel une opération devra se produire
# Ici l'utilisation du where permet de spécifier les conditions d'exécution 

# La fonction d'ajout add() additionne le contenu de deux tab et renvoie les resultats
arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])
add_2listes = np.add(arr1, arr2)
print("Addition de deux listes:", newarr)

# La fonction de soustraction subtract() soustrait le contenu de deux tab et renvoie les resultats
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])
soustrac_2Liste = np.subtract(arr1, arr2)
print("Soustraction de deux listes:", soustrac_2Liste)

# La fonction de multiplication multiply() multiplie le contenu de deux tab et renvoie les resultats
multiply_2Listes = np.multiply(arr1, arr2)
print("Miltiplication de deux listes:", multiply_2Listes)

# La fonction de division divide() divise le contenu de deux tab et renvoie les resultats
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 5, 10, 8, 2, 33])
divise_2Listes = np.divide(arr1, arr2)
print("Division de 2 listes:", divise_2Listes)

# La fonction puissance power() met à la puissance le contenu d'un tableau par rapport à un autre et renvoie les resultats
puissance_entre2Listes = np.power(arr1, arr2)
print("Permet de mettre à la puissance 2 Listes:", puissance_entre2Listes)

# Reste de la division via mod() et remainder() qui renvoie le reste de la division entre 2 listes 
modulo_entre2Listes = np.mod(arr1, arr2)
modulo_entre2Listes1 = np.remainder(arr1, arr2)
print("Reste de la division:", modulo_entre2Listes, modulo_entre2Listes1)

# La fonction qui renvoie à la fois le quotient et modulo divmod() retourne deux tableau 1 pour le quotient et l'autre pour le modulo
quotAndModulo_entre2Listes = np.divmod(arr1, arr2)
print("Renvoie à la fois le quotient et modulo dans deux tableaux", quotAndModulo_entre2Listes)

# La fonction valeur Absolue via absolute() effectue des opérations de valeur absolue
arr = np.array([-1, -2, 1, 2, 3, -4])
valeurAbsolue = np.absolute(arr)
print("Opérations de valeur absolue:", valeurAbsolue)
#------------------------------------------------------------------------------------------

# ------------------------ Arrondir les décimales dans les ufunc --------------------------
# 5 façon permettent d'arrondir les decimales:

# troncature supprime les décimale et renvoie un nombre floattant point zero via les fonction trunc()et fix()
troncature = np.trunc([-3.1666, 3.6667])
print("troncature sur des nombres floattant", troncature)               # resultat [-3.  3.] 

fix_onFloat = np.fix([-3.1666, 3.6667])
print("Fix équivalent de troncature sur les floattant:", fix_onFloat)   # resultat [-3.  3.]

# L'arrondi via around() incremente de 1 si les la decimale est >=5 exple: (3.16 ==> 3.2)
arrondi = np.around(3.1666, 2)
print("Arrondi sur les chiffres floatants:", arrondi)

# Arrondi decimale de l'entier inférieur le plus proche via floor()
arrondi_decimalInf = np.floor([-3.1666, 3.6667])
print("Arrondi décimale de l'entier inférieur:", arrondi_decimalInf)     # resultat [-4.  3.]

# Arrondit la décimale à l'entier supérieur le plus proche via ceil()
arrondi_decimalSup = np.ceil([-3.1666, 3.6667])
print("Arrondi décimale de l'entier supérieur:", arrondi_decimalSup)      # resultat [-3.  4.]
#------------------------------------------------------------------------------------------

# ------------------------ Les journaux(log) dans Numpy ufunc --------------------------
# Trouver log de (base e) ou log (naturel) de tous les éléments du tableau
mylist = np.arange(1, 10)  
log_base = np.log(mylist)
print("Trouver log de (base e) ou log (naturel) de tous les éléments du tableau", log_base)

# Trouver le journal(log) en base 2 de tous les éléments du tableau
mylist = np.arange(1, 10)           # fonction renvoie une liste de nbre de valeur allant de 1 à 9 
log_base2 = np.log2(mylist)
print("Trouver le journal(log) en base 2 de tous les éléments du tableau", log_base2)

# Trouver le journal en base 10 de tous les éléments du tableau 
mylist = np.arange(1, 10)           # fonction renvoie une liste de nbre de valeur allant de 1 à 9 
log_base10 = np.log10(mylist)
print("Trouver le journal(log) en base 10 de tous les éléments du tableau", log_base10)

# Numpy ne fournissant aucun log nous utiliserons frompyfunc() et math.log() pour créer une fonction 
# log qui prend en entrée 2 paramètres et en sortie un paramètre 
nplog = np.frompyfunc(log, 2, 1)
log_naturel= nplog(100, 15)
print("Log naturel:", log_naturel)        # resultat 1.7005483074552052
#------------------------------------------------------------------------------------------

# ------------------------------ La somme dans Numpy ufunc --------------------------------
# La différence entre la somme et l'addition
# L'addition se fait entre deux arguments
# La somme se fait sur (n) éléments

# addition
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
add_Liste = np.add(arr1, arr2)
print("Addition se fait sur 2 arg", add_Liste)        # Résultat un tableau [2 4 6]

# Somme de n elements
sum_N_element = np.sum([arr1, arr2])
print("Somme de n elements", sum_N_element)         # Résultat un chiffre "12"

# somme cumulative consiste à ajouter partiellement les du tableau via la fonction cumsum()
# Exemple [1, 2, 3, 4] serait [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
maListe = np.array([1, 2, 3])
sommeCumulative = np.cumsum(maListe)
print("somme cumulative:", sommeCumulative)           # resultat [1 3 6]

# somme sur un axe pour cella nous devons spécifier axis, Numpy additionnera les nombres de chaque tableau
sommation_surAxe = np.sum([arr1, arr2], axis=1)
print("sommation sur un axe", sommation_surAxe)         # resultat [6 6]
#------------------------------------------------------------------------------------------

# ------------------------------ La somme dans Numpy ufunc --------------------------------
# Trouver le produit des éléments de ce tableau via la fonction prod()
arr = np.array([1, 2, 3, 4])
produit_tab = np.prod(arr)
print("Le produit des éléments d'un tableau", produit_tab)        # resultat un chiffre 1*2*3*4 = 24

# Trouver le produit des éléments de deux tableaux
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
produit_de2Listes = np.prod([arr1, arr2])
print("Produit des éléments de deux tableaux", produit_de2Listes)   #Resultat un chiffre 1*2*3*4*5*6*7*8 = 40320

# Produit sur un axe, via l'axis
produit_de2Liste_Axis = np.prod([arr1, arr2], axis=1)
print("Produit sur un axe avec axis=1:", produit_de2Liste_Axis)     #Resultat est un tableau  [  24 1680]

# Produit cumulatif via la fonction cumprod()  xple: [1, 2, 3, 4] est [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]
arr = np.array([5, 6, 7, 8])
produit_cumulatif = np.cumprod(arr)
print("Produit cumulatif:", newarr)           #Resultat est un tableau  [5 30 210 1680]
#------------------------------------------------------------------------------------------

# ------------------------------ La différence dans Numpy ufunc --------------------------------
# Une différence discrète diff() signifie soustraire deux éléments successifs
# pour [1, 2, 3, 4], la différence discrète serait [2-1, 3-2, 4-3] = [1, 1, 1]

# Calculez la différence discrète du tableau suivant
arr = np.array([10, 15, 25, 5])
difference_discrete = np.diff(arr)
print("La différence discrète d'une liste", difference_discrete)       #Resultat est un tableau [  5  10 -20]

# Calculez (n=2) fois la différence discrète du tableau suivant
difference_discrete = np.diff(arr, n=2)                                # On calcul une 1er fois puis une 2nd fois 
print("La différence discrète d'une liste", difference_discrete)       # Resultat est un tableau [  5  10 -20] puis [  5 -30]
#------------------------------------------------------------------------------------------

# ---------------------- Le plus petit commun multiple ufunc ------------------------------
# c'est le plus petit commun multiple est le plus petit nombre commun multiple de deux nombres
# Elle se calcul avec la fonction lcm()

# Trouvez le LCM des deux nombres suivants 
num1 = 4
num2 = 6
ppcm = np.lcm(num1, num2)
print("Le plus petit commun multiple est", ppcm)            # resultat est un chiffre 12 car (4*3=12 et 6*2=12)

# La methode reduce() permet de trouver le PPCM de toutes les valeurs d'un tableau ou liste
arr = np.array([3, 6, 9])
PPCM_surTableau = np.lcm.reduce(arr)
print("le PPCM de toutes les valeurs d'un tableau:", PPCM_surTableau)        # resultat est un chiffre 18 car (3*6=18, 6*3=18 et 9*2=18).

# Trouvez le LCM de toutes les valeurs d'un tableau où le tableau contient tous les entiers de 1 à 10
arr = np.arange(1, 11)
PPCM_liste = np.lcm.reduce(arr)
print("le LCM de toutes les valeurs d'un tableau où le tableau contient tous les entiers de 1 à 10", PPCM_liste)
#------------------------------------------------------------------------------------------

# ----------------- Le plus grand denominateur commun multiple ufunc ------------------------------
# C'est le plus grand nombre qui est un facteur commun aux deux nombres

# Trouvez le PGCM des deux nombres suivants
num1 = 6
num2 = 9
PGDC_nbre = np.gcd(num1, num2)
print("le PGDC des deux nombres suivants:", PGDC_nbre)        # resultat est un chiffre 3 

# Trouvez le PGCD pour tous les nombres dans le tableau suivant via gcd() et le reduce())
arr = np.array([20, 8, 32, 36, 16])
PGCD_tab = np.gcd.reduce(arr)
print("le PGCD pour tous les nombres dans le tableau", PGCD_tab)          # resultat est un chiffre 4
#-------------------------------------------------------------------------------------------------

# ---------------------------------- fonction trigonométrique ufunc ------------------------------
# convertir les dégré en radiant 
arr = np.array([90, 180, 270, 360])
arr_1 = np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])

degre_to_radian = np.deg2rad(arr)
degre_to_radianTab = np.rad2deg(arr)
print("Convertir les dégrés en radiant:", degre_to_radian, degre_to_radianTab)      # resultat est un tableau [1.57079633 3.14159265 4.71238898 6.28318531]

# Trouver la valeur sinusoïdale de PI/2
sinusoide = np.sin(np.pi/2)
print("La valeur sinusoïdale de PI/2:", sinusoide)

# Trouvez les valeurs sinus pour toutes les valeurs du tableau
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
sinus_vlrTab = np.sin(arr)
print(" les valeurs sinus pour toutes les valeurs du tableau:", sinus_vlrTab)

# Pour trouver les valeurs des angles nous utiliserons arcsin(), arccos()et arctan()qui produisent 
# des valeurs en radian pour les valeurs sin, cos et tan 

# Trouvez l'angle de 1.0
determine_angle = np.arcsin(1.0)
print("l'angle de 1.0:", determine_angle)                   # l'angle en radian 1.57079

# Trouver l'angle pour toutes les valeurs sinus dans le tableau
arr = np.array([1, -1, 0.1])
angle_prVlrTab = np.arcsin(arr)
print("l'angle pour toutes les valeurs sinus dans le tableau", angle_prVlrTab)

# Calcul de l'hypoyténues d'un triangle via la fonction hypot()
# Trouvez les hypoténués pour 4 bases et 3 perpendiculaires
base = 3
perp = 4
hypothenus = np.hypot(base, perp)
print("L'hypoténués pour 4 bases et 3 perpendiculaires", hypothenus)      # l'angle en radian 5.0
#-------------------------------------------------------------------------------------------------

# ---------------------------------- Fonctions hyperboliques NumPy ------------------------------
# Pour calculer des hyperbolles dans NumPy nous utiliserons sinh(), cosh()et tanh()qui prennent des valeurs en radians
# produisent les valeurs sinh, cosh et tanh correspondante

# Trouver la valeur sinh de PI/2
sin_hyperbolique = np.sinh(np.pi/2)
print("la valeur sinh de PI/2 est:", sin_hyperbolique)

# Trouvez les valeurs de cosh pour toutes les valeurs du tableau
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
cosh_vlrTab = np.cosh(arr)
print("Les valeurs de cosh pour toutes les valeurs du tableau", cosh_vlrTab)

# pour trouver les angles nous utiliserons arcsinh(), arccosh() et arctanh()qui produisent des valeurs en radian
recherche_angle = np.arcsinh(1.0)
print("trouver les angles:", recherche_angle)

# Trouvez l'angle pour toutes les valeurs de tanh dans le tableau
arr = np.array([0.1, 0.2, 0.5])
angle_tab = np.arctanh(arr)
print("angles de chaque valeurs d'un tableau:", angle_tab)
#-------------------------------------------------------------------------------------------------

# ---------------------------------- Opration sur les ensembles NumPy ------------------------------
# Trouver une différence symétrique via setxor1d()
# La différence symétrique permet trouver uniquement les valeurs qui ne sont PAS présentes dans les DEUX ensembles

# Trouvez la différence symétrique de set1 et set2
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])
diff_symétrique = np.setxor1d(set1, set2, assume_unique=True)
print("La différence symétrique de set1 et set2", diff_symétrique)       # resultat est un tableau [1 2 5 6]

# La difference consiste  rechercher uniquement les valeurs du premier ensemble qui ne sont PAS
#  présentes dans l'ensemble des seconds. via setdiff1d()

# Trouvez la différence entre l'ensemble1 et l'ensemble2
difference = np.setdiff1d(set1, set2, assume_unique=True)
print("La différence entre l'ensemble1 et l'ensemble2", difference)       # resultat est un tableau [1 2]

# L'intersection permet de rechercher uniquement les valeurs 
# présentes dans les deux tableaux, via intersect1d()

# Trouver l'intersection des deux ensembles de tableaux suivants
intersection_2tab = np.intersect1d(arr1, arr2, assume_unique=True)
print("l'intersection des deux ensembles de tableaux:", intersection_2tab)
#-------------------------------------------------------------------------------------------------





print(tab_numpy)
print(type(tab_numpy))
print(numpy_version)
print(tab_0_D)
print(tab_1_D)
print(tab_2D)
print(tab_3D)
print("dimension du tab:", tab_xD.ndim)
print(dim_tab)
print()

print("1er element tab 1D:", first_elem)
print("1er element tab 2D:", first_elem_2D)
print("1er element de la 1ere liste du tab_2D:", first_elem_2D_ofFisrtList)
print("3ème élément du 2ème tableau du 1er tableau:", thirdElement_of_2ndTab_of_firstTab)
print()

print("decoupage index 1 à l'index 5 du tab_1D:", decoup_1par5)
print("decoupage de l'index 2 à la fin du tab_1D:", decoup_2parfin)
print("decoupage du début à l'index 4 du tab_1D:", decoup_debutpar4)
print("decoupage index -3 à l'index -1  du tab_1D:", decoupNegtf_neg3Parneg1)
print("decoupage index 1 à l'index 4  plus marche de 2 du tab_1D:", marche_1Par4par2)
print("decoupage par par la marche de 2 du tab_1D:", marche_Parpar2)
print("decoupez le 2ème élément, découpez de l'index 1 à l'index 4 du tab2D:", decoup2D_1par4)
print("À partir des deux éléments, retournez l'index 2 du tab2D:", decoup2D_2Elmtparindex3)
print("À partir des deux éléments, retournez de l'index 1 par index 3 du tab2D:", decoup2D_2Elmt1par3)

print()
print("Verifier le type d\'un array:", dtype_array)
print("vérification du type d'une liste de chaine:", type_listChaine)
print("creation d'un tab de chaine avec dtype:", tab_chaine)
print("Type d'un tableau de chaine:", type_tabChaine)
print("creation d'un tab d'entier de 4 octect avec dtype:", tab_int_4octect)
print("Type d'un tab d'entier de 4 octect:", type_tabInt4oct)
print("Tab de float:", tab_of_floatt)
print("Type du nouveau tab de float convertit en entier:", type_tabfloaot_toInt)
print("convertion tab float en tab entier:", tab_int)
print("Type tab float en tab entier:", type_tabInt)

print()
print("Tableau d'origine modifier à son premier index:", arr)
print("Copie du tableau d'origine:", x)
print("Creation du View du tableau d'origine:", y)
print("Tableau d'origine modifier à son premier index:", arr)
print("Vérifie si la copie possède ses données:", have_dataCopy)
print("Vérifie si la vue possède ses données:", have_dataView)

print()
print("shape du tableau :", forme_tap2D)
print("shape du tableau :", forme_tap2D_ndmin)

print()
print("Reshape du tableau 1D à tableau 2D :", redim_tap1D_toTap2D)
print("Vérifions si le tableau redimensionné à ses propres données:", have_dataRedim_tap1D_toTap2D)
print("Table 1D vers tableau dimension inconnu: ", redim_tap1D_toTapinconnu)
print("Applatir un tableau 2D en un tableau 1D ", applat_tab2D_toTab1D)
#print(type(tab_2_D))