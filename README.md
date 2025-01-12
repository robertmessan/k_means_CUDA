# k_means_CUDA

## Projet du cours parallel programing avec CUDA: Télécom SudParis

Dans ce projet nous avons choisi d'implémenter l'algorithme de Kmeans sur CUDA. Le test de l'algorithme a
été réalisé avec le jeu de données CIFAR-10 qui contient 60000 images labellisées et classées en 10 catégories.
Chaque image est en couleur et de format 32 X 32 pixels. Représentées donc suivant 3 channels, la dimension d'un seul point de notre dataset est : 3*32*32 = 3072. 

## Test des hyperparamètres du modèle Kmeans et des paramètres du programme CUDA

Pour les hyperparamètres du modèle des kmeans, nous distinguons le nombre de clusters et la dimension
d'un point du jeu de donnée(une image). En pratique, on devrait faire des tests pour trouver le nombre
de clusters, mais l'objectif n'étant pas d'avoir un clustering parfait, nous allons donc choisir 10 parce
qu'on a 10 clusters(dans un cas réel, on trouve k empiriquement) et faire la comparaison avec l'implémentation
sur CPU pour les mêmes hyperparamètres.
Nous avons du modifier les paramètres du programme CUDA taille des blocs et de la grille manuellement. Des différences remarquables ont été observées en modifiant la taille des blocs. En effet, en diminuant la taille
les performances chutent.

## Exécution des deux codes:

Un fichier Makefile permet de générer les exécutables pour les deux codes principaux. Il suffit d'exécuter les commandes suivantes:
*cmake .* et *make*

Trois exécutables sont donc générés et peuvent être exécutés : 
*./kmean_par* ou *./shared_kmeans_par* ou alors *./kmean_seq*
respectivement pour le code *CUDA*, *CUDA optimisé avec shared memory et coalescing* et *CPU*

## Réalisé par Kossi Robert M. 
## Implémentation inspirée du projet *thrust* de *NVIDIA*, métrique de *Davies-Bouldin index* 