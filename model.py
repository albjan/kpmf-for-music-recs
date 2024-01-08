import math
from typing import Callable, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def default(x, y):
    return 1 if x == y else 0


def euclidean(x, y):
    return np.dot(x, y) / 10


def rbf(x, y):
    return np.exp(-1/2 * np.linalg.norm(x - y)**2)


class MusicModel:
    '''
    A trainable kernelized probabilistic matrix factorization model.
    '''
    def __init__(
        self, 
        n_artists: int, n_tags: int, k: int, 
        kernel_a: Optional[str] = 'default',
        kernel_u: Optional[str] = 'default', 
        s_u: float = 1, s_a: float = 1, s_r: float = 1, tt_split: float = 0.8
    ):
        if kernel_a not in {'default', 'euclidean', 'rbf'}:
            raise ValueError(
                'Only supported artist kernels are: default, euclidean, rbf'
            )
        if kernel_u not in {'default', 'ct'}:
            raise ValueError(
                'Only supported artist kernels are: default, ct'
            )   
        self.kernel_a = kernel_a
        self.kernel_u = kernel_u
        self.m = n_artists
        self.n_tags = n_tags 
        self.k = k
        self.s_u = s_u
        self.s_a = s_a 
        self.s_r = s_r
        self.tt_split = tt_split

        self._preprocess()
        self._initialize_latents()
        self._form_cov_matrices()
        self._initialize_grad_vars()

    def _preprocess(self):
        self._filter_and_map_data()
        self._split_train_test()

    def _initialize_latents(self):
        self.U = torch.rand(self.n, self.k, requires_grad=True)
        self.V = torch.rand(self.m, self.k, requires_grad=True)

    def _filter_and_map_data(self):
        
        artist_cnts = ratings['artistID'].value_counts()
        self.top_n_artists = artist_cnts.head(self.m).index.tolist()
        self.aid_to_i = { # maps artist id -> i 
            aid: i for i, aid in enumerate(self.top_n_artists)
        }

        self.uids = ratings['userID'].unique().tolist()
        self.uid_to_i = { # maps user id -> i
            uid: i for i, uid in enumerate(self.uids)
        }
        self.n = len(self.uids)

        tag_cnts = user_tagged['tagID'].value_counts() 
        self.top_n_tags = tag_cnts.head(self.n_tags).index.tolist()
        self.tid_to_i = { # maps tag id -> cov matrix i 
            tid: i for i, tid in enumerate(self.top_n_tags)
        }

        self.user_artist_tags = user_tagged[['artistID', 'tagID']].loc[
            (user_tagged['artistID'].isin(self.top_n_artists))
            & (user_tagged['tagID'].isin(self.top_n_tags))
        ]
        self.user_artist_tags['artistID'] = self.user_artist_tags['artistID'].map(self.aid_to_i)
        self.user_artist_tags['tagID'] = self.user_artist_tags['tagID'].map(self.tid_to_i)

        self.ratings = ratings.loc[ratings['artistID'].isin(self.top_n_artists)]
        self.ratings['rating'] = np.log(self.ratings['weight'])
        self.ratings['userID'] = self.ratings['userID'].map(self.uid_to_i)
        self.ratings['artistID'] = self.ratings['artistID'].map(self.aid_to_i)

        self.user_friends = user_friends.copy()
        self.user_friends['userID'] = self.user_friends['userID'].map(self.uid_to_i)
        self.user_friends['friendID'] = self.user_friends['friendID'].map(self.uid_to_i) 

    def _form_cov_matrices(self):
        self._form_artist_cov()
        self._form_user_cov()

    def _form_artist_embeddings(self):
        # self.artist_embeddings[i][j] = indicator[artist i was given tag j]
        self.artist_embeddings = np.zeros(shape=(self.m, self.n_tags))
        for _, row in self.user_artist_tags.iterrows(): 
            a_id, t_id = row['artistID'], row['tagID']
            self.artist_embeddings[a_id][t_id] = 1


    def _form_artist_cov(self):
        self._form_artist_embeddings()
        if self.kernel_a == 'default':
            self.C_a = torch.eye(self.m) * self.s_a**2
            return
        
        kernel_func = {
            'euclidean': euclidean,
            'rbf': rbf,
        }

        self.C_a = torch.zeros(self.m, self.m)
        for i in range(self.m):
            for j in range(self.m):
                self.C_a[i][j] = kernel_func[self.kernel_a](
                    self.artist_embeddings[i],
                    self.artist_embeddings[j]
                )

    def _form_user_embeddings(self):
        pass
        
    def _form_user_cov(self):
        self._form_user_embeddings()
        if self.kernel_u == 'default':
            self.C_u = torch.eye(self.n) * self.s_u**2
            return

        if self.kernel_u == 'ct':
            self.adj = torch.zeros(self.n, self.n)
            for _, row in self.user_friends.iterrows():
                u1_id, u2_id = row['userID'].astype(int), row['friendID'].astype(int)
                self.adj[u1_id, u2_id] = 1
                self.adj[u2_id, u1_id] = 1
            self.D = torch.diag(torch.sum(self.adj, axis=1))
            self.L = self.D - self.adj
            self.C_u = 1 / torch.pinverse(self.L) # CT kernel matrix

    def _split_train_test(self):
        self.train_user_rated = defaultdict(set) # user u_id -> set of artists (a_id) rated
        self.train_artist_rated_by = defaultdict(set) # artist (a_id) -> set of users (u_id) who rated that artist
        self.test_user_rated = defaultdict(set)
        self.test_artist_rated_by = defaultdict(set) 
        self.rating_pairs = {}

        self.ratings = self.ratings.sample(frac=1) # shuffle 
        self.train_sz = int(self.ratings.shape[0] * self.tt_split)
        self.train_ratings = self.ratings.iloc[:self.train_sz, :]
        self.test_ratings = self.ratings.iloc[self.train_sz:, :]

        for _, row in self.train_ratings.iterrows():
            u_id, a_id, rating = row['userID'].astype(int), row['artistID'].astype(int), row['rating']
            self.train_user_rated[u_id].add(a_id)
            self.train_artist_rated_by[a_id].add(u_id)
            self.rating_pairs[(u_id, a_id)] = rating

        for _, row in self.test_ratings.iterrows():
            u_id, a_id, rating = row['userID'].astype(int), row['artistID'].astype(int), row['rating']
            self.test_user_rated[u_id].add(a_id)
            self.test_artist_rated_by[a_id].add(u_id)
            self.rating_pairs[(u_id, a_id)] = rating

    def _initialize_grad_vars(self):
        self.C_a_inv = torch.inverse(self.C_a)
        self.C_u_inv = torch.inverse(self.C_u)

        self.mask_train = torch.zeros(self.n, self.m) # I[i, j] = 1[user i rated artist j in train], serves as mask
        self.mask_test = torch.zeros(self.n, self.m)
        self.R_train = torch.zeros(self.n, self.m)
        for uid, artists in self.train_user_rated.items():
            for aid in artists:
                self.R_train[uid, aid] = self.rating_pairs[(uid, aid)]
                self.mask_train[uid, aid] = 1

        self.R_test = torch.zeros(self.n, self.m)
        for uid, artists in self.test_user_rated.items():
            for aid in artists:
                self.R_test[uid, aid] = self.rating_pairs[(uid, aid)]
                self.mask_test[uid, aid] = 1

    def _get_train_loss(self):
        u_reg = 1/2 * torch.trace(self.U.T @ self.C_u_inv @ self.U)
        v_reg = 1/2 * torch.trace(self.V.T @ self.C_a_inv @ self.V) 
        
        ratings_loss = (
            1 / (2 * self.s_r**2) *
            torch.norm(
                (self.U @ self.V.T - self.R_train) *
                self.mask_train
            )**2
        )
        
        return u_reg + v_reg + ratings_loss

    def get_posterior_predictive(self):
        return (
            -1 / (2 * self.s_r**2) *
            torch.norm(
                (self.U @ self.V.T - self.R_test) *
                self.mask_test
            )**2
        )

    def get_rmse(self):
        return torch.sqrt(
            1 / self.test_ratings.shape[0] *
            torch.norm(
                (self.U @ self.V.T - self.R_test) *
                self.mask_test
            )**2   
        ) 

    def _do_grad_desc(self, n_iters, lr, threshold):
        for epoch in range(n_iters):
            loss = self._get_train_loss()
            prev_loss = loss
            loss.backward()

            with torch.no_grad():
                self.U -= lr * self.U.grad
                self.V -= lr * self.V.grad
                self.U.grad.zero_()
                self.U.grad.zero_()

            # if epoch % 50 == 0:
            #     print(f'iter: {epoch}')
            #     print(f'loss: {loss}')

            loss = self._get_train_loss()
            if (abs(loss - prev_loss) < threshold):
                print(f'converged on iter {epoch}')
                break

    def train(self, n_iters=1000, lr=1e-4, threshold=1e-4):
        self._initialize_latents()
        self._do_grad_desc(n_iters, lr, threshold)