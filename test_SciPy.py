import scipy
from scipy import constants
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.optimize import root
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.csgraph import bellman_ford
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import cosine
from scipy.spatial.distance import hamming
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import Rbf
from scipy.stats import kstest
from scipy.stats import ttest_ind
from scipy import io
import matplotlib.pyplot as plt
from math import cos
import numpy as np


print(constants.liter)
print(scipy.__version__)
print()


# --------------------------------- Python Scientifique  -----------------------------------
# SciPy pour python scientifique c'est une bibliothèque pour le calcul scientifique.
# Elle utilise NumPy en dessous, bibliothèque optimisée et est open source.
# cette bibliothèque Python fournit des fonctions utilitaire pour:
# (Optimisation, les statisque, traitement de signal)

#-------------------------------------------------------------------------------------------

# -------------------- Les constantes dans (SciPy) Python Scientifique  ----------------------
# SciPy étant axé sur le calcul scientifique il fournit en interne de nombreuses constantes scientifques
# Ces constantes seront utilise par exple lors d'un travail en Data science exple: PI

# Imprimer la valeur de la constante de PI :
valeur_pi = constants.pi
print("La valeur de la constante de PI:", valeur_pi)

# Pour Lister toutes les constantes internes à SciPy
Liste_toutesConstantes_scipy = dir(constants)
print("Lister toutes les constantes internes à SciPy:", Liste_toutesConstantes_scipy)
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes métriques dans (SciPy) ----------------------------
# unité renvoyé dans métrique est le metre exple: centi renvoie 0.01

print(constants.yotta)    #1e+24
print(constants.zetta)    #1e+21
print(constants.exa)      #1e+18
print(constants.peta)     #1000000000000000.0
print(constants.tera)     #1000000000000.0
print(constants.giga)     #1000000000.0
print(constants.mega)     #1000000.0
print(constants.kilo)     #1000.0
print(constants.hecto)    #100.0
print(constants.deka)     #10.0
print(constants.deci)     #0.1
print(constants.centi)    #0.01
print(constants.milli)    #0.001
print(constants.micro)    #1e-06
print(constants.nano)     #1e-09
print(constants.pico)     #1e-12
print(constants.femto)    #1e-15
print(constants.atto)     #1e-18
print(constants.zepto)    #1e-21
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes du binaires dans (SciPy) ----------------------------
# unité renvoyé est l' Octect exple: kibi renvoie 1024 Octects

print(constants.kibi)    #1024
print(constants.mebi)    #1048576
print(constants.gibi)    #1073741824
print(constants.tebi)    #1099511627776
print(constants.pebi)    #1125899906842624
print(constants.exbi)    #1152921504606846976
print(constants.zebi)    #1180591620717411303424
print(constants.yobi)    #1208925819614629174706176
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de la Masse dans (SciPy) ----------------------------
# unité renvoyé est le Kg(kilogramme) exple: gram renvoie 0.001 Kg

print(constants.gram)        #0.001
print(constants.metric_ton)  #1000.0
print(constants.grain)       #6.479891e-05
print(constants.lb)          #0.45359236999999997
print(constants.pound)       #0.45359236999999997
print(constants.oz)          #0.028349523124999998
print(constants.ounce)       #0.028349523124999998
print(constants.stone)       #6.3502931799999995
print(constants.long_ton)    #1016.0469088
print(constants.short_ton)   #907.1847399999999
print(constants.troy_ounce)  #0.031103476799999998
print(constants.troy_pound)  #0.37324172159999996
print(constants.carat)       #0.0002
print(constants.atomic_mass) #1.66053904e-27
print(constants.m_u)         #1.66053904e-27
print(constants.u)           #1.66053904e-27
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes des angles  dans (SciPy) ----------------------------
# unité renvoyé est le radian exple: degree renvoie 0.017453292519943295 redian

print(constants.degree)     #0.017453292519943295
print(constants.arcmin)     #0.0002908882086657216
print(constants.arcminute)  #0.0002908882086657216
print(constants.arcsec)     #4.84813681109536e-06
print(constants.arcsecond)  #4.84813681109536e-06
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes du temps dans (SciPy) ----------------------------
# unité renvoyé est la seconde exemple, hour renvoie 3600.0 secondes

print(constants.minute)      #60.0
print(constants.hour)        #3600.0
print(constants.day)         #86400.0
print(constants.week)        #604800.0
print(constants.year)        #31536000.0
print(constants.Julian_year) #31557600.0
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de la longueur dans (SciPy) ----------------------------
# unité renvoyé est le mètre exemple nautical_mile renvoie 1852.0 mètres

print(constants.inch)              #0.0254
print(constants.foot)              #0.30479999999999996
print(constants.yard)              #0.9143999999999999
print(constants.mile)              #1609.3439999999998
print(constants.mil)               #2.5399999999999997e-05
print(constants.pt)                #0.00035277777777777776
print(constants.point)             #0.00035277777777777776
print(constants.survey_foot)       #0.3048006096012192
print(constants.survey_mile)       #1609.3472186944373
print(constants.nautical_mile)     #1852.0
print(constants.fermi)             #1e-15
print(constants.angstrom)          #1e-10
print(constants.micron)            #1e-06
print(constants.au)                #149597870691.0
print(constants.astronomical_unit) #149597870691.0
print(constants.light_year)        #9460730472580800.0
print(constants.parsec)            #3.0856775813057292e+16
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de la pression dans (SciPy) ----------------------------
# unité renvoyé est le pascals exemple, psirenvoie 6894.757293168361 pascals

print(constants.atm)         #101325.0
print(constants.atmosphere)  #101325.0
print(constants.bar)         #100000.0
print(constants.torr)        #133.32236842105263
print(constants.mmHg)        #133.32236842105263
print(constants.psi)         #6894.757293168361
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de la zone dans (SciPy) ----------------------------
# unité renvoyé est le mètres carrés exemple, hectare renvoie 10000.0 mètres carrés

print(constants.hectare) #10000.0
print(constants.acre)    #4046.8564223999992
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes du Volume dans (SciPy) ----------------------------
# unité renvoyée  est mètres cubes exemple, literrenvoie 0.001 mètres cubes

print(constants.liter)            #0.001
print(constants.litre)            #0.001
print(constants.gallon)           #0.0037854117839999997
print(constants.gallon_US)        #0.0037854117839999997
print(constants.gallon_imp)       #0.00454609
print(constants.fluid_ounce)      #2.9573529562499998e-05
print(constants.fluid_ounce_US)   #2.9573529562499998e-05
print(constants.fluid_ounce_imp)  #2.84130625e-05
print(constants.barrel)           #0.15898729492799998
print(constants.bbl)              #0.15898729492799998
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de la vitesse dans (SciPy) ----------------------------
# unité renvoyé est le mètres par seconde exemple, speed_of_sound renvoie 340.5 mètres par seconde

print(constants.kmh)            #0.2777777777777778
print(constants.mph)            #0.44703999999999994
print(constants.mach)           #340.5
print(constants.speed_of_sound) #340.5
print(constants.knot)           #0.5144444444444445
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de la température dans (SciPy) ----------------------------
# unité renvoyé est Kelvin exemple zero_Celsius renvoie 273.15 Kelvin

print(constants.zero_Celsius)      #273.15
print(constants.degree_Fahrenheit) #0.5555555555555556
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de l'energie dans (SciPy) ----------------------------
# unité renvoyé est joules exemple, calorie renvoie 4.184 joules

print(constants.eV)            #1.6021766208e-19
print(constants.electron_volt) #1.6021766208e-19
print(constants.calorie)       #4.184
print(constants.calorie_th)    #4.184
print(constants.calorie_IT)    #4.1868
print(constants.erg)           #1e-07
print(constants.Btu)           #1055.05585262
print(constants.Btu_IT)        #1055.05585262
print(constants.Btu_th)        #1054.3502644888888
print(constants.ton_TNT)       #4184000000.0
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes de la puissance dans (SciPy) ----------------------------
# unité renvoyé est le  watts exemple, horsepower renvoie 745.6998715822701 watts

print(constants.hp)         #745.6998715822701
print(constants.horsepower) #745.6998715822701
#---------------------------------------------------------------------------------------------

# ------------------------- Les constantes  de la force dans (SciPy) ----------------------------
# unité renvoyé est le newton exemple, kilogram_force renvoie 9.80665 newton

print(constants.dyn)             #1e-05
print(constants.dyne)            #1e-05
print(constants.lbf)             #4.4482216152605
print(constants.pound_force)     #4.4482216152605
print(constants.kgf)             #9.80665
print(constants.kilogram_force)  #9.80665
#---------------------------------------------------------------------------------------------

# --------------------------------- Optimiseurs dans (SciPy) ---------------------------------
# Sont un ensemble de procédure dans scipy qui permettent de trouver soit (la valeur minimale), 
# soit la (racine d'une équation).

# La racine d'une équation 
# Numpy est capable de trouver les racines des polynomes et des équation lineaire.
# (optimze.root) permet de trouver les racines des équations non linéaires, prend deux arguments:
# (l'équation fun) qui représente une équation
# (racine initale x0) qui est une estimation initiale de la racine 
# La solution retournée par l'équation est un objet dont l'attribut (x) est la solution réelle

# Trouver la racine de l'équation x + cos(x)
def eqn(x):
    return x + cos(x)
      
racine_equt_nonLineaire = root(eqn, 0)            # fun = eqn et x0 = 0
valeurX_in_RacinEqn = racine_equt_nonLineaire.x
print("la racine de l'équation x + cos(x): \n", racine_equt_nonLineaire)
print("Impression de la valeur de x dans la solution:", valeurX_in_RacinEqn)

## Minimiser une fonction
# Dans ce  contexte la fonction est une courbe avec :
# des point haut maxima
# des point bas minima
# le (maxima global) est le point le (plus haut d'une courbe) et les autre maxima locaux
# Le minima global est le point le plus bas de et les autres sont minima locaux

## Trouver les minima d'une équation
# a fonction scipy.optimize.minimize() permet de minimiser une fonction
# minimize() prends les arguments suivants:
# (fun): une equation 
# (x0): une estimation initiale pour la racine
# (method): nom de la methode à utiliser {'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'}
# (callback): fonction appelée après chaque itération d'optimisation
# (options): un dictionnaire définissant des paramètres supplémentaires 
"""{
     "disp": boolean - print detailed description
     "gtol": number - the tolerance of the error
  }"""

# Minimisez la fonction x^2 + x + 2 avec BFGS:
def eqn(x):
    return x**2 + x + 2

mymin = minimize(eqn, 0, method='BFGS')
print("Minimistion de la fonction x^2 + x + 2 avec BFGS: \n", mymin)
#---------------------------------------------------------------------------------------------

# ----------------------------- Données fragmentées dans (SciPy) -----------------------------
# Les données fragmentées sont des données qui contiennent des éléments inutilisés ou 
# données ne contenant aucune information
# données éparses: ensemble de données dans lequel la plupart des valeurs d'éléments sont égales à zéro.

# Pour traiter des données éparses nous utiliserons le module (scipy.sparse) qui fournit des fonctions
# Nous utiliserons deux types de matrices creuses:
# (CSC: colonne claisemée compressée) Pour une arithmétique efficace et un découpage rapide des colonnes.
# (CSR: ligne claisemée compressée) Pour un découpage rapide des lignes, des produits vectoriels matriciels plus rapides


## Matrice RSE
# La creation de matrice CSR se fait en passant un tableau à la fonction scipy.sparse.csr_matrix()
# Créez une matrice CSR à partir d'un tableau
arr = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2])
matrice_csr = csr_matrix(arr)
print("Création d'une matrice CSR à partir d'un tableau: \n", matrice_csr)

## Méthodes de matrice creuse, les non-zéros, Suppression des entrées nulles
# Affichage des données stockées (pas les éléments zéro) avec la propriété (data)
arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
matrice_creuse = matrice_csr.data
print("Matrice creuse ne cotenant aucune données egale à zero:", matrice_creuse)

# pour compter les éléments non égale à zéro on utilise la méthode (count_nonzero())
element_non_equalZero = matrice_csr.count_nonzero()
print("Les éléments non égale à zéro:", element_non_equalZero)

# Suppression des entrées nulles de la matrice avec la méthode (eliminate_zeros())
suppression_entreeNull = matrice_csr.eliminate_zeros()
print("Suppression des entrées nulles de la matrice:", suppression_entreeNull)

# Élimination des entrées en double avec la méthode (sum_duplicates())
suprression_entreeDouble = matrice_csr.sum_duplicates()
print("Élimination des entrées en double:", suprression_entreeDouble)

# Conversion de (csr) en (csc) avec la méthode tocsc()
convert_csr_to_csc = matrice_csr.tocsc()
print("Conversion de (csr) en (csc): \n", convert_csr_to_csc)
#---------------------------------------------------------------------------------------------

# ----------------------------- Les graphiques dans (SciPy) ----------------------------------
# les graphique sont une structure essentielle, le module scipy.sparse.csgraph permet de travailler sur ces données

## Les matrices contiguit
# Une matrice adjacence est une matrice  (n x n) où :
# (n): est le nombre déléments dans un graphe
# (les valeurs): représente la connexion entre les éléments.
"""
exempl de matrice adjacence
   R :[0 1 2]  
   B :[1 0 0]
   C :[2 0 0]
"""

## Composant connectées 
# Pour déterminer tous les composants connectés on utilise la méthode connected_components()
arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])

matrice_csr = csr_matrix(arr)
composant_connecte = connected_components(matrice_csr)
print("Détermination de tous les composants connectés:", composant_connecte)

## Dijkstra
# Permet de déterminer le chemin le plus court dans un graphe d'un point à un autre
# Prend les arguments suivants:
# (return_predecessors): booléen (True pour renvoyer le chemin complet de la traversée sinon False)
# (indices): index de l'élément pour renvoyer tous les chemins à partir de cet élément uniquement
# (limit): poids maximal du chemin

chemin_plusCourt_via_djikstra = dijkstra(matrice_csr, return_predecessors=True, indices=0)
print("Déterminer le chemin le plus court dans un graphe:", chemin_plusCourt_via_djikstra)

## Floyd Warhall
# Permet de determiner le chemin le plus court entre toutes les paires d'éléments via floyd_warshall()
chemin_plusCourt_entrePaire_viaFloydWarhall = floyd_warshall(matrice_csr, return_predecessors=True)
print("le chemin le plus court entre toutes les paires d'éléments:", chemin_plusCourt_entrePaire_viaFloydWarhall)

## Groom Ford
# la methode bellman_ford() Permet de determiner le chemin le plus court entre toutes les paires d'éléments
# Et peut également gérer les poids négatifs.
bellmanFord = bellman_ford(matrice_csr, return_predecessors=True, indices=0)
print("le chemin le plus court de l'élément 1 à 2 avec un graphique donné avec un poids négatif:", bellmanFord)

## Profondeur premier ordre 
# Elle renvoie à partir d'un noeud le premier parcours en profondeur via la méhode depth_first_order()
# Cette fonction prend les arguments suivants :
# le graphique.
# l'élément de départ à partir duquel parcourir le graphe.

arr = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [2, 1, 1, 0], [0, 1, 0, 1]])
matrice_csr = csr_matrix(arr)
ponfondeur_first_ordre = depth_first_order(matrice_csr, 1)
print("Parcourez la profondeur du graphe pour une matrice d'adjacence donnée:", ponfondeur_first_ordre)

## Largeur premier ordre
# Cette méthode  breadth_first_order() renvoie un premier parcours en largeur à partir d'un noeud
# Cette fonction prend les arguments suivants :
# le graphique.
# l'élément de départ à partir duquel parcourir le graphe.
matrice_csr = csr_matrix(arr)
largeur_graphe_mtrxContiguit = breadth_first_order(matrice_csr, 1)
print("Parcourez la largeur du graphe pour une matrice de contiguïté donnée:", largeur_graphe_mtrxContiguit)
#---------------------------------------------------------------------------------------------

# ----------------------------- Les données spatials dans (SciPy) ----------------------------------
# Ce sont les données représentées dans un espace géométrique exple des points dans un système de coordonées
# Elle permet de réaliser des tâches tels que (trouver si un point est à l'intérieur d'une frontière ou non)
# Pour le faire nous utiliserons le module (scipy.spatial) qui possède des fonctions nous permettant de travailler 
# sur les données spatiales


### La triangulation
# La triangulation d'un polygone: est une méthode qui consiste à diviser un polygone en plusieurs triangles avec lequels 
# on peut calculer l'aire de polygone.
# La triangulation avec des points: consiste à créer des triangles composés de surface dans lesquels tous les points donnés
# sont sur au moins un sommets de n'importe quel triangle de la surface
# on utilisera la méthode (Delaunay())

# Création d'une triangulation à partir des points suivants:
tab_2D = np.array([[2, 4],[3, 4],[3, 0],[2, 2],[4, 1]])
simplices = Delaunay(tab_2D).simplices            # la propriété (simplices): permet de généraliser la notation triangulaire

plt.triplot(tab_2D[:, 0], tab_2D[:, 1], simplices)
plt.scatter(tab_2D[:, 0], tab_2D[:, 1], color='r')
plt.show()

### Enveloppe convexe
# C'est le plus petit polygone qui couvre tous les points donnés.
# On utilise la méthode (ConvexHull()) pour déterminer l'enveloppe convexe

# Création d'une enveloppe convexe pour les points suivants :
tab_2d = np.array([[2, 4], [3, 4], [3, 0], [2, 2], [4, 1], [1, 2], [5, 0], [3, 1], [1, 2], [0, 2]])

hull = ConvexHull(tab_2d)
hull_points = hull.simplices
plt.scatter(tab_2d[:,0], tab_2d[:,1])

for simplex in hull_points:
  plt.plot(tab_2d[simplex, 0], tab_2d[simplex, 1], 'k-')

plt.show()

### KDTreesGenericName
# est une tructure de données optimisée pour les requêtes du plus proche voisin
# Exple: Rechercher des points les plus proche d'un point donné.
# Pour cela nous utiliserons la méthode KDTree() qui renvoie un objet KDTree
# La méthode (query()) renvoie la distance du voisin le plus proche et son emplacement.

# Trouvez le voisin le plus proche du point (1,1):
points = [(1, -1), (2, 3), (-2, 3), (2, -3)]

kdtree = KDTree(points)                # Rechercher des points les plus proche du point(1, 1)
resultat = kdtree.query((1, 1))        # renvoie la distance du voisin le plus proche et l'emplacement

print("Trouvez le voisin le plus proche du point (1,1):", resultat)

### Matrice des distances
# Ilexiste de nombreuses façon de trouver ou de calculer la distances entre 2 points en sciences de donnée 
# on a la distance euclidienne, cosinus, etc
# La distance entre 2 vecteur peut etre:
# La longeur de la ligne droite entre 2 points
# nombre de pas unitaire requis
# l'angle entre eux depuis l'origine.

### Distance euclidienne entre des points données
p1 = (1, 0)
p2 = (10, 2)

distance_euclidienne = euclidean(p1, p2)
print("Distance euclidienne entre des points données P1 et P2:", distance_euclidienne)

### Distance du pâté de maisons (distance de Manhattan)
# Cette distance est calculée en utilisant 4 degrés de mouvement
p1 = (1, 0)
p2 = (10, 2)

dstnc_pate_maison = cityblock(p1, p2)
print("Distance du pâté de maisons:", dstnc_pate_maison)

### Cosinus Distance
# C'est la valeur de l'angle cosinus entre les deux points A et B
p1 = (1, 0)
p2 = (10, 2)

cosinus_dstnce = cosine(p1, p2)
print("Cosinus Distance entre des points donnés:", cosinus_dstnce)

### Distance de Hamming
# Est la proportion de bits où deux bits sont différents
# Utilisé pour calculer la distance pour les séquences binaires.

# Trouver la distance de Hamming entre des points donnés
p1 = (True, False, True)
p2 = (False, True, True)

distance_hamming = hamming(p1, p2)
print("La distance de Hamming entre des points donnés P1 et P2:", distance_hamming)
#---------------------------------------------------------------------------------------------

# ----------------------------- Les tableaux (SciPy) Matlab ----------------------------------
# Le module (scipy.io) fournit des fonctions pour travailler avec les ableaux matlab

### Exportation de données au format Matlab
# La fonction savemat() permet d'exporter des données au format Matlab
# Elle prend les paramètres suivants:
# (filename) - le nom du fichier pour enregistrer les données exple: (nom_ficier.mat)
# (mdict) - un dictionnaire contenant les données exple: {"keys_element_stocker": element_a_stocker}
# (do_compression) - une valeur booléenne qui spécifie s'il faut ou non compresser le résultat. Faux par défaut

#Exportez le tableau suivant sous le nom de variable "vec" dans un fichier mat:
arr = np.arange(10)
io.savemat('arr.mat', {"vec": arr})

### Importer des données à partir du format Matlab
# La fonction loadmat() permet d'importer des données à partir d'un fichier Matlab
# La fonction prend un paramètre obligatoire :
# (filename) - le nom de fichier des données enregistrées

# Importez le tableau à partir du fichier mat suivant :.
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,])

# Export file:
io.savemat('arr.mat', {"vec": arr})

# Import file:
mydata = io.loadmat('arr.mat')
# affiche le contenu du fichier
print("Importez le tableau à partir du fichier mat:", mydata)
# afficher juste un tableau contenu dans le fichier
print("afficher uniquement le tableau des données matlab:", mydata["vec"])

# Pour conserver les dimensions d'un tableau d'origine nous utiliserons (squeeze_me=True)
conserv_tab_origin = io.loadmat('arr.mat', squeeze_me=True)
print("Conserver les dimensions d'un tableau d'origine:", conserv_tab_origin['vec'])
#---------------------------------------------------------------------------------------------

# ----------------------------- Interpolation  dans (SciPy) ----------------------------------
# C'est une méthode qui consiste à générer des points entre des points donnés.
# Elle permet par exple dans le machine learning de générer des données manquantes par interpolation
# On parle (imputation) qui est cette methode de remplissage de valeur.
# Elle est egalement utilisée pour lisser des points discrets pour un ensemble de données 
# Le module (scipy.interpolate) fournit un ensemble de fonction pour gérer l'interpolation
# ici les points sont ajustés sur une seule courbe.


### Interpolation 1D
# La fonction (interp1d()) permet d'interpoler une distribution à 1 variable
# Pour xs et ys donnés, interpolez les valeurs de 2.1, 2.2... à 2.9 :
xs = np.arange(10)
ys = 2*xs + 1

interp_func = interp1d(xs, ys)                      # Interpolation 1D
newarr = interp_func(np.arange(2.1, 3, 0.1))
print("Pour xs et ys donnés, interpolez les valeurs de 2.1, 2.2... à 2.9:", newarr)

### Interpolation spline
# Dans cette interpolation les points sont ajustés par rapport à une fonction par morceaux définie 
# avec des polynômes appelés splines
# La fonction UnivariateSpline()fonction prend xsand ys et produit une fonction appelable qui peut être appelée avec new xs.
# Fonction par morceaux : une fonction qui a une définition différente pour différentes plages.

#Trouver l'interpolation spline univariée pour 2.1, 2.2... 2.9 pour les points non linéaires
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1

interp_func = UnivariateSpline(xs, ys)              # Interpolation spline
newarr = interp_func(np.arange(2.1, 3, 0.1))
print("Trouver l'interpolation spline univariée pour 2.1, 2.2... 2.9 pour les points non linéaires:", newarr)

### Interpolation avec fonction de base radiale
# La fonction de base radiale est une fonction définie correspondant à un point de référence fixe
# La fonction (Rbf()) prend également xset ys comme arguments et produit 
# une fonction appelable qui peut être appelée avec new xs

# Interpolez les xs et ys suivants en utilisant rbf et trouvez les valeurs pour 2.1, 2.2 ... 2.9 :
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1

interp_func = Rbf(xs, ys)                           # Interpolation avec fonction de base radiale
newarr = interp_func(np.arange(2.1, 3, 0.1))
print("En utilisant rbf, trouvez les valeurs pour 2.1, 2.2 ... 2.9 :", newarr)
#---------------------------------------------------------------------------------------------

# -------------------- Tests de signification statistique dans (SciPy) ------------------------
# La signification statistique signifie que le résultat qui a été produit a une raison derrière ce resultat 
# Ce resultat n'est pas produit au hasard ou par hasard.
# Le module (scipy.stats) à des fonctions qui effectuent des tests de signification statistique
# Technique de signification statistique:
# (Hypothèse en statistique): est une hypothèse sur un paramètre de la population
# (Hypothèse nulle): suppose que l'observation n'est pas statistiquement significative

## Test unilatéral
# Lorsque notre hypothèse teste un seul côté de la valeur

## Test bilatéral
# Lorsque notre hypothèse teste les deux côtés des valeurs.

## Test T (un test bilatéral)
# Sont utilisés pour déterminer s'il existe une différence significative entre les moyennes de deux variables
# Et nous permet de savoir si elles appartiennent à la même distribution.
# Via (ttest_ind()) prend deux échantillons de même taille et produit un (tuple de statistique t et de valeur p)

## Valeur alpha
# La valeur alpha est le niveau de signification

## Valeur P
# Indique à quel point les données sont proches de l'extrême 
# La (valeur P) et les (valeurs alpha) sont comparées pour établir la signification statistique
# Si la valeur (p <= alpha), nous rejetons l'hypothèse nulle et disons que les données sont statistiquement significatives.

# Déterminez si les valeurs v1 et v2 données proviennent de la même distribution :
v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)

res = ttest_ind(v1, v2)                 # Test T via ttest_ind() qui est un test bilatéral
print("Déterminez si les valeurs v1 et v2 données proviennent de la même distribution:", res)

# Pour renvoyer uniquement la valeur on utilise 
p_valeur = ttest_ind(v1, v2).pvalue
print("Déterminez si les valeurs v1 et v2 données proviennent de la même distribution en affichant uniquement pvalue:", p_valeur)

## Test KS 
# (considéré comme test unilatéral ou bilatéral, par défaut il est considéré comme bilatéral)
# ce test est utilisé pour vérifier si des valeurs données suivent une distribution.
# elle prend deux paramètres:
# la valeur à tester
# la CDF peut être (une chaîne) ou (une fonction) appelable qui renvoie la probabilité

# Déterminez si la valeur donnée suit la distribution normale:
v = np.random.normal(size=100)
resultat_test_ks = kstest(v, 'norm')
print("Déterminez si la valeur donnée suit la distribution normale:", resultat_test_ks)
#-----------------------------------------------------------------------------------------------