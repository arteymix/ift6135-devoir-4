---
bibliography: bib.json
---

Pour reconstruire des images de bonne qualité, nous nous sommes inspirés de
l'architecture VGG-16 [@] et des techniques utilisés pour DCGAN
[@http://zotero.org/users/3733213/items/3WJPTN3T]. En particulier, nous avons
utilisé une activation Tanh et un rescaling pour reconstruire le spectre RGB du
générateur.

Couche      Détails
------      -------
conv2d      32 kernel $3 \times 3$
conv2d      32 kernel $3 \times 3$ strides 2
max pooling strides 2
conv2d      64 kernel $3 \times 3$
conv2d      64 kernel $3 \times 3$ strides 2
max pooling strides 2
flatten

: Architecture de l'encodeur

Le décodeur suivant l'architecture suivante:

Couche     Détails
------     -------
dense      $W_{100 \times 16 386} + b_{16 384}$
reshape    $16 384 \rightarrow 16 \times 16 \times 64$
upscaling  implementation-dependant
deconv2d   32 kernel $3 times 3$
upscaling  implementation-dependant
deconv2d   32 kernel $3 \times 3$

: Architecture du décodeur

L'upscaling utilisé dépend du type de décodeur:

 - déconvolution striée avec kernel $3 \times 3$ strides 2
 - interpolation du plus-proche-voisin avec facteur 2
 - interpolation bilinéaire avec facteur 2

Nous avons empilé une 2 convolution avec 1 max pooling.

Nous avons également essayé la normalisation par lot pour accélérer
l'entraînement, mais avons noté que cette approche avait tendance à corrompre
les images produites par le décodeur.

Nous avons comparés les trois approches suivantes:

 - déconvolutions striées
 - upscaling par le plus proche voisin
 - upscaling par interpolation bilinéaire

Nous avons intégré l'interpolation du plus proche voisin[^resize_neighbor] et
bilinéaire[^resize_bilinear] de Tensorflow. La déconvolution striée était déjà
implémentée dans Keras.

[^resize_nearest]: https://www.tensorflow.org/api_docs/python/tf/image/resize_nearest
[^resize_bilinear]: https://www.tensorflow.org/api_docs/python/tf/image/resize_bilinear

Les trois modèles ont été entraînés sur l'ensemble d'entraînement avec un split
de 33% pour la validation. Le modèle de la meilleure époque a été conservé pour
l'évaluation des images générées.

