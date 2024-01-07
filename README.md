# Probabilistic Models (STCS 6701) Final Project

## Title: Kernelizing Probabilistic Matrix Factorization to Enhance Probabilistic Matrix Factorization

### Abstract

We apply **kernelized probabilistic matrix factorization** (KPMF) to model the rating a user assigns a music artist in the `hetrec2011-lastfm-2k` dataset. KPMF introduces complexity to probabilistic matrix factorization (PMF) by capturing covariances between the latent vectors learned, rather than assuming they are i.i.d. 

We construct two covariance matrices — one for users and one for artists — a priori using side information provided by the dataset, and incorporate them into the generative process. 

- To construct the user covariance matrix, we leverage the social network graph provided and apply the *Commute Time (CT) graph kernel*, which embeds nodes (users) in an inner product space.
- To construct the artist covariance matrix, we embed each artist with a vector that stores tags assigned to that artist by users, and then apply the *Radial Basis Function (RBF) kernel* on two artists' embeddings to obtain their covariance.

We learn the latent vectors via **MAP inference**. The CT kernel and RBF kernel both yield slightly lower RMSEs than PMF, but combining the two is minimally helpful.
