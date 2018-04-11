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

Ok!
