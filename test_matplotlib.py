
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib est une bibliothèque de traçage de graphes de bas niveau 
# en python qui sert d'utilitaire de visualisation

"""
# ---------------------------------------- Pyplot -----------------------------------------
# La plupart des utilitaires Matplotlib se trouve sous pyplot qui est un sous module de Matplotlib

# Tracez une ligne dans un diagramme de la position (0,0) à la position (6,250)

xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

plt.plot(xpoints, ypoints)
plt.show()
#------------------------------------------------------------------------------------------

# ----------------------------------- Tracé Matplotlib ------------------------------------
# La fonction plot() permet de tracer des points dans un diagramme 
# Par défaut elle trace une ligne d'un point à un autre, elle prend comme arguments les
# paramètres (axe des X et axe des Y) pour une ligne de (1, 3) à (8, 10)nous devons passer à plot
# à plot nous lui envoyons le tableau des x [1,8] et tab des Y [3,10]

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
# tracé avec ligne 
plt.plot(xpoints, ypoints)
plt.show()

# pour un tracé sans ligne via l'aargument 'o'
plt.plot(xpoints, ypoints, 'o')
plt.show()

# Pour un tracé multiple ou pour des points multiples 
# il faudrait disposer d'un meme nombre de point sur les deux axes
# une ligne dans un diagramme de la position (1, 3) à (2, 8) puis à (6, 1) et enfin à la position (8, 10)
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
# tracé avec ligne 
plt.plot(xpoints, ypoints)
plt.show()

# Par défaut si nous ne spécifions pas les valeurs de X sur l'axe des X il prend la valeur 1,2,3,4 etc
# donc si nous omettons les point X notre diagramme resemblera à ceci avec nos points en y
ypoints = np.array([3, 8, 1, 10, 5, 7])

# Des points x seront attribués par défauts et auront les valeurs [0, 1, 2, 3, 4, 5]
plt.plot(ypoints)
plt.show()
#------------------------------------------------------------------------------------------

# ----------------------------------- Marqueurs Matplotlib ------------------------------------
# pour souligner chaque point avec un marqueur spécifique on utilise l'argument marker
# Marquons chaque point avec un cercle et une étoile

# marqueurs avec étoile sur les points
plt.plot(ypoints, marker = '*')
plt.show()

# marqueurs avec cercle sur les points 
plt.plot(ypoints, marker = 'o')
plt.show()

# le formatage des chaines ou (des lignes de jonction des points) fmt
# Appelé fmt il a comme syntaxe (marker|line|color)
plt.plot(ypoints, 'o-.g')
plt.show()

# taille du marqueur est l'argument markersize ou ms pour définir la taille du marqueur 
plt.plot(ypoints, marker = 'o', ms = 20)
plt.show()

# Pour la couleur (des bords) du marqueur on uilise l'arg (markeredgecolorou le plus court mec)
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')
plt.show()

# Pour la couleur intérieur des et des bors on utilise l'arg (markerfacecolor ou le plus court mfc)
plt.plot(ypoints, marker = 'o', ms = 20, mfc = 'r')
plt.show()

# Utilisation à la fois de la couleur des bords et de la couleur intérieur des marqueur.
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'g', mfc = 'r')
plt.show()

# Nous pouvons également utiliser les valeurs des couleurs hexadécimale
plt.plot(ypoints, marker = 'o', ms = 20, mec = '#4CAF50', mfc = '#4CAF50')
plt.show()

# Ou l'un des 140 noms pris en charge par matplotlib (OrangeRed, OliveDrab, Navy, SlateBlue, Yellow, SpringGreen)
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'hotpink', mfc = 'hotpink')
plt.show()
#------------------------------------------------------------------------------------------


# ----------------------------------- Marqueurs Matplotlib ------------------------------------
# Pour modifier le style des ligne tracées on peut utiliser (linestyle, ou short ls)

# Pour des lignes en pointillée
plt.plot(ypoints, linestyle = 'dotted')
plt.plot(ypoints, linestyle = 'dashed')
plt.show()

# style des lignes dans une syntaxe plus courte via (ls = '')
plt.plot(ypoints, ls = ':')
plt.show()

# la couleur de la ligne se fait via l'argument (color ou le plus court c)
# ou via l'hexadécimal de la couleur ou meme les couleurs prises en charge par plot()
plt.plot(ypoints, c = '#4CAF50')
plt.plot(ypoints, color = 'r')
plt.plot(ypoints, c = 'hotpink')
plt.show()

# La largeur des lignes se faiit via l'argument (linewidthou le plus court lw)
plt.plot(ypoints, linewidth = '10.5')
plt.show()

# tracer plusieurs lignes en ajoutant autant de plt.plot(y1) que l'on souhaite
y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])

plt.plot(y1)
plt.plot(y2)
plt.show()

# Tracer plusieurs ligne en ajoutant autant les x1, y1 sur la même ligne
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 8, 1, 10])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 7, 11])

plt.plot(x1, y1, x2, y2)
plt.show()
#------------------------------------------------------------------------------------------

# --------------------------- Étiquettes et titre Matplotlib -------------------------------
# Pour définir des étiquettes dans Pyplot sur l'axe des X et des Y on utilise:
#  xlabel() et ylabel()

# Ajoutez des libellés aux axes x et y
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)
plt.xlabel("Average X Pulse")
plt.ylabel("Calorie Y Burnage")
plt.show()

# Création des titres pour un tracé se fait la fonction title()
# Ajoutez un titre de tracé et des étiquettes pour les axes x et y
plt.plot(x, y)

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.show()

# Definition des polices pour le titre et les étiquettes via le paramètre fontdict
# fontdict est utilisé en tant que paramètre dans xlabel(), ylabel() et title()

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.plot(x, y)
plt.show()

# Position du titre se fait via le paramètre loc dans title()
# Positionnez le titre à gauche :
plt.title("Sports Watch Data", loc = 'left', fontdict = font1)
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.show()
#------------------------------------------------------------------------------------------

# ---------------------- Ajout de lignes de grille dans Matplotlib -------------------------
# L'ajout de grille dans matplotlib se fait via la fonction grid() ce qui 
# permet d'ajouter une grille au tracé matplotlib

# Ajoutez des lignes de grille au tracé :
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.plot(x, y)
plt.grid()              # Ajout de grille au tracé 
plt.show()

# Spécification sur les grilles à ajouter, Pour spécifier les grille à afficher 
# on peut les valeur 'x','y' et 'both'
# 'both': affiche à la fois la grille de l'axe des X et des Y

# Afficher uniquement les lignes de grille pour l'axe des x
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.grid(axis = 'x')                 # (axis = 'x'): Affiche la grille pour l'axe des x
plt.show()

# Afficher uniquement les lignes de grille pour l'axe des ordonnées
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.grid(axis = 'y')                # (axis = 'y'): Affiche la grille pour l'axe des y
plt.show()

# Définition des propriétés de la grille 
# La definition des propriétés de la grilles se fait de cette manière
# grid(color = ' color ', linestyle = ' linestyle ', linewidth = number )

# Définissez les propriétés des lignes de la grille :
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)        # def propriété des lignes de la grille
plt.show()
#-------------------------------------------------------------------------------------------

# ---------------------- Les sous-parcelles dans Matplotlib --------------------------------
# pour désinner plusieurs tracés dans une figure nous utilisons subplot() qui prend 3 arguments:
# Le 1er et 2ème arguments represente la mise en page
# le 3ème arguments représente l'index du tracé courant

# Dessinez 2 parcelles :
#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)            # Affiche des 2 graphes sur les lignes ou (1 ligne, 2 colonnes et 2 position du graphe)
plt.plot(x,y)

plt.show()

# Ajout de titre à chaque tracé  et d'un super titre pour tous les tracé
# L'ajout d'un super titre pour tous les tracés se fait via suptitle()
# L'ajout de titre à chaque tracé se fait via title()

#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.title("SALES")                  # Ajout de titre à chaque tracé

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("INCOME")                  # Ajout de titre à chaque tracé

plt.suptitle("MY SHOP")              # Ajout du supertitre à chaque tracé
plt.show()
#-------------------------------------------------------------------------------------------
"""
# ---------------------- Scatter(nuage des points) Matplotlib ------------------------------
# Pour dessiner un nuage de point nous pouvons utiliser la fonction scatter()
# il prends en arguments 2 tableaux de même longueur 

## Un simple nuage de points :
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)           # Création d'un nuage de points grâce à leur cordonnées (x,y)
plt.show()

## Comparons deux tracés 
# Et personnalisation avec nos propres couleurs via le paramètre color 
# Exple: color = 'hotpink'

#day one, the age and speed of 13 cars:
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y, color = 'hotpink')        # Personnalisation de la couleur des points de notre nuage

#day two, the age and speed of 15 cars:
x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y, color = '#88c999')        # Personnalisation de la couleur des points de notre nuage

plt.show()

## Pour colorier chaque point 
# On crée un tableau de couleurs égale au nombre de points 
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])

plt.scatter(x, y, c=colors)
plt.show()

### Combinaison de la taille et des couleurs 
# Utilisation d'une palette de couleur se fait via l'argument (cmap) et utilise la palette de couleur 
# intégré dans Matplotlib est appelée 'viridis'
# Affichage de la platte de couleur avec la fonction colorbar()
# Def de la taille des points avec le paramètre (s= tab_sizes)

# Créez un tableau de couleurs et spécifiez une palette de couleurs dans le nuage de points
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])

plt.scatter(x, y, s=sizes, c=colors, cmap='nipy_spectral')    
plt.colorbar()          # Affichage de la palette de couleur
plt.show()

# Alpha: ajuster la transparence des point avec l'argument alpha 
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])

plt.scatter(x, y, s=sizes, alpha=0.5)       # Ajuster la transparence
plt.show()
#-------------------------------------------------------------------------------------------

# ---------------------------- les barres dans Matplotlib ----------------------------------
# La fonction bar() permet de créer des barres dans Matplotlib

## Dessinez 4 barres 
x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)                    # Permet de dessiner des barres
plt.show()

## Dessinez 4 barres Horizontales
plt.barh(x, y)                  # barh permet de dessiner des barres horizontales
plt.show()

# Définition des couleur des barres avec l'argument color
# valeur des coleur peu etre un string ("red") ou valeur hexadecimale ("#4CAF50")
plt.bar(x, y, color = "red")            # barre verticale de couleur rouge
plt.show()

## La largeur des barres avec l'argument width
# Dessinez 4 barres très fines
plt.bar(x, y, width = 0.1)              # Barre fine 
plt.show()

## La hauteur des barres avec l'argument height
# Dessinez 4 barres très fines
plt.barh(x, y, height = 0.1)            # barre horizontale fine avec barh
plt.show()
#-------------------------------------------------------------------------------------------

# ---------------------------- les barres dans Matplotlib ----------------------------------
# la fonction hist() dans Matplotlib permet de créer des histagrammes
# hist prend en argument un tableau qu'il utilise pour créer un histogramme

# Un histogramme simple
x = np.random.normal(170, 10, 250)

plt.hist(x)
plt.show() 
#-------------------------------------------------------------------------------------------

# ---------------- Création de graphiques à secteurs dans Matplotlib ------------------------
# La fonction pie() permet de dessiner des graphiques à secteurs style camemberts

# Un graphique circulaire simple 
y = np.array([35, 25, 25, 15])

plt.pie(y)
plt.show() 

# Ajout des étiquettes avec le paramètre labels
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(y, labels = mylabels)
plt.show()

# Ajout une legende la fonction legend()
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(y, labels = mylabels)
plt.legend()                 # Permet d'ajouter une legende au graphe en secteur 
plt.show() 

# Ajout de legende avec en tête ou titre  
plt.pie(y, labels = mylabels)
plt.legend(title = "Four Fruits:")
plt.show() 

# Pour démarquer l'un des coin du secteur on utilisera explode 
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0.2, 0, 0, 0]                                   # definition du secteur exposé 

plt.pie(y, labels = mylabels, explode = myexplode)
plt.show() 

# Ajout des ombres avec le paramètre shodow qui est une valeur booleen (true ou false)
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
myexplode = [0.2, 0, 0, 0]

plt.pie(y, labels = mylabels, explode = myexplode, shadow = True)
plt.show() 

# Définition des couleur pour chaque secteur via le paramètre color 
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
mycolors = ["black", "hotpink", "b", "#4CAF50"]

plt.pie(y, labels = mylabels, colors = mycolors)
plt.show()
#--------------------------------------------------------------------------------------------

version_matplolib = matplotlib.__version__
print("Version matplotlib sur ce PC:", version_matplolib)


