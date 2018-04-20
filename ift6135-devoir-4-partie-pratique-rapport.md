---
title: IFT6135 --- Devoir 4
authors:
 - Guillaume Poirier-Morency
 - Augustin Schmidt
bibliography: bib.json
---

# Generating Faces

Nous avons traité les images avec skimage en appliquant un bruit $Uniform(0,1)$
et un rescaling entre $[0, 1]$ afin de pouvoir considérer les canaux comme des
probabilités d'émission.

![](figures/preprocessing-color-distribution-histogram.png)

Nous remarquons un léger biais vers les valeurs de saturation des canaux
\ref{figure:1}, ce qui est expliqué par des régions particulièrement sombre des
images.

# Model

Nous avons implanté l'auto-encodeur variationel.

Pour reconstruire des images de bonne qualité, nous nous sommes inspirés de
l'architecture VGG-16 [@http://zotero.org/users/3733213/items/NU4NX8HZ] et des
techniques utilisés pour DCGAN [@http://zotero.org/users/3733213/items/3WJPTN3T].
En particulier, nous avons utilisé une activation sigmoïde pour reconstruire le
spectre RGB du générateur normalisé sur l'intervalle $[0, 1]$.

Couche      Détails
------      -------
conv2d      32 kernel $3 \times 3$
conv2d      32 kernel $3 \times 3$ strides 2
max pooling strides 2
conv2d      64 kernel $3 \times 3$
conv2d      64 kernel $3 \times 3$ strides 2
max pooling strides 2
flatten
dense       100 avec activation Tanh

: Architecture de l'encodeur \label{table:1}

L'encodeur utilise une activation Tanh telle que décrite dans le tableau
\ref{table:1}. Nous avons eu des problèmes avec l'activation ReLU qui faisait
parfois exploser la variance de l'inférence variatonelle.

Le décodeur possède l'architecture générale suivante:

Couche     Détails
------     -------
dense      $W_{100 \times 16 386} + b_{16 384}$
reshape    $16 384 \rightarrow 16 \times 16 \times 64$
upsampling dépend de l'implémentation
deconv2d   32 kernel $3 times 3$
upsampling dépend de l'implémentation
deconv2d   3 kernel $3 \times 3$

: Architecture du décodeur

L'upsampling utilisé dépend du type de décodeur:

 - déconvolution striée avec kernel $3 \times 3$ strides 2
 - interpolation du plus-proche-voisin avec facteur 2 et déconvolution 2d avec
 kernel $3 \times 3$
 - interpolation bilinéaire avec facteur 2 et déconvolution 2d avec
 kernel $3 \times 3$

Nous avons également essayé la normalisation par lot pour accélérer
l'entraînement, mais avons noté que cette approche avait tendance à corrompre
les images produites par le décodeur. Par conséquent, nous l'avons seulement
appliqué sur l'encodeur.

Nous avons comparés les trois approches suivantes:

 - déconvolutions striées
 - upsampling par le plus proche voisin
 - upsampling par interpolation bilinéaire

Nous avons intégré l'interpolation du plus proche voisin[^resize_neighbor] et
bilinéaire[^resize_bilinear] de Tensorflow. La déconvolution striée était déjà
implémentée dans Keras.

[^resize_nearest]: https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest
[^resize_bilinear]: https://www.tensorflow.org/api_docs/python/tf/image/resize_bilinear

Les trois modèles ont été entraînés sur l'ensemble d'entraînement avec un split
de 33% pour la validation. Le modèle de la meilleure époque a été conservé pour
l'évaluation des images générées.

