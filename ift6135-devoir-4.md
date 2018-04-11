---
title: IFT6135 Devoir 4
author: Guillaume Poirier-Morency <guillaume.poirier-morency@umontreal.ca>
header-includes:
 - \DeclareMathOperator{\expected}{\mathbb{E}}
 - \DeclareMathOperator{\var}{\mathbb{V}ar}
---

# Reparameterization Trick of Variational Autoencoder

## a)

\begin{align*}
\expected[\mu(x) + \sigma(x) \odot \epsilon] \\
&= \expected[\mu(x)] + \expected[\sigma(x) \odot \epsilon] & \mu(x) \bot \sigma(x) \text{ pour les distributions Gaussiennes} \\
&= \expected[\mu(x)] + \sigma(x) \odot \expected[\epsilon] \\
&= \mu(x)
\end{align*}

\begin{align*}
\var[\mu(x) + \sigma(x) \odot \epsilon] \\
&= \var[\sigma(x) \odot \epsilon] \\
&= \sigma(x)^2 \odot \var[\epsilon] \\
&= \sigma(x)^2 & \epsilon \text{ a une variance unitaire }
\end{align*}

Par conséquent, $z \sim \mathcal{N}(\mu(x), \sigma(x)^2)$.

\begin{align*}
\expected[\mu(x) + S(x) \epsilon] \\
&= \expected[\mu(x)] + S(x)\expected[\epsilon] \\
&= \mu(x)
\end{align*}

\begin{align*}
\var[\mu(x) + S(x)\epsilon] \\
&= \var[\mu(x)] + \var[S(x)\epsilon] \\
&= S(x)\var[\epsilon]S(x)^T \\
&= S(x)S^T(x)
\end{align*}

Par conséquent, $z \sim \mathcal{N}(\mu(x), S(x)S(x)^T)$

## b)

Le modèle $q^{mf}$ permet de modéliser l'espace latent par un ensemble de
distributions présumées indépendantes. L'avantage du VAE est que l'inférence
est beaucoup moins coûteuse: un modèle (i.e. réseau de neurones) s'occupe
d'inférer la distribution $p(z_i \mid x_i)$ et ne fait pas nécessairement la
présomption d'indépendance. En particulier, on pourrait estimer une matrice de
covariance complète d'une Gaussienne multivariée.

# Importance Weighted Autoencoder

## a)

\begin{align*}
\mathcal{L}_k &= \expected_{x \sim p(x)} \left[ \frac{1}{k} \sum_{i=1}^k \log \frac{p(x,z_i)}{p(z_i \mid x)} \right] \\
&\leq \frac{1}{k} \sum_{i=1}^k \log \expected_{x \sim p(x)} \left[ \frac{p(x,z_i)}{p(z_i \mid x)} \right] & \text{par concavité du } \log \\
&= \frac{1}{k}\sum_{i=1}^k \log p(x) \\
&= \log p(x)
\end{align*}

## b)

\begin{align*}
\mathcal{L}_2 &= \expected_{x \sim p(x)} \left[\log \frac{1}{2}\sum_{i=1}^2 \frac{p(x,z_i)}{p(z_i|x)} \right] \\
&> \expected_{x \sim p(x)} [\frac{1}{2} \sum_{i=1}^2 \log \frac{p(x,z_i)}{p(z_i|x)}] & \text{par concavité stricte du } \log \\
&=  \frac{1}{2} \sum_{i=1}^2 \expected_{x \sim p(x)} \left[ \log \frac{p(x,z_i)}{p(z_i|x)} \right] \\
&= \mathcal{L}_1
\end{align*}

Puisque $\mathcal{L}_1 < \mathcal{L}_2 \leq p(x)$, il constitue une borne
inférieur plus stricte.

# Maximum Likelyhood for Generative Adversarial Networks

On cherche à trouver les paramètres $\theta$ qui maximisent la vraisemblance
des exemplaires produits par le GAN selon la vraie distribution des données
$P_{data}$:

\begin{align}
\max_\theta \expected_{z\sim p(z)}[\log P_{data}(G_\theta(z))]
\end{align}

\begin{align*}
\max_\theta \expected_{z \sim p(z)} f(D^*(G_\theta(z))) \\
&= \max_\theta \expected_{z \sim p(z)} \left[ f(D^*(G_\theta(z))) \right] \\
&= \max_\theta \expected_{z \sim p(z)} \left[ f(\frac{P_{data}(G_\theta(z))}{P_{data}(G_\theta(z)) + P_{gen}(G_\theta(z))}) \right]
\end{align*}

Ici, on permet au GAN de récupérer le $z$ correspondant à l'exemplaire $x$
généré puisque c'est son objectif. Une fonction $f$ optimale selon le maximum
de vraisemblance serait:

\begin{align}
f(x, z) = \log x + \log (P_{data}(G_\theta(z)) + P_{gen}(G_\theta(z)))
\end{align}
