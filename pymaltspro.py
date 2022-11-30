#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 10 2022

@author: harshparikh and kattasa
"""
import numpy as np
import scipy.optimize as opt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings
# from scipy.statistics import wasserstein_distance as wd
warnings.filterwarnings("ignore")
import ot


def wasserstein_dist(sample_array1, sample_array2, p, n_samples_min, array1_quantile = False, array2_quantile = False):
    '''
    description
    -----------
    calculate pairwise wasserstein distance between all the sampled distributions of one array and all the sampled distributions of another
    Wasserstein distance for dists U, V on real line: 
        int_{0}^1 (F^{-1}_U(q) - F^{-1}_V(q)) dq, where F^{-1}_A reps quantile function of rv A

    inputs
    ------
    sample_array1: N x S_max1 np array of floats/ints
        N is number of units in the array and S_max1 is the maximum number of samples across any of the N dists
    sample_array2: N x S_max2 np array of floats/ints
        N is number of units in the array and S_max2 is the maximum number of samples across any of the N dists
    p: order of wasserstein distance

    returns
    -------
    Wasserstein-p distance between sample_array1 and sample_array2
    '''
    # if the point is scalar, evaluate only at the 1st quantile
    if n_samples_min == 1:
        quantile_values = np.array([1])
        quantile_diffs = [1]
    else:
        quantile_values = np.linspace(start = 0, stop = 1, num = n_samples_min)
        # find width of each quantile window
        quantile_diffs = quantile_values[1:] - quantile_values[:-1]
        quantile_diffs = np.hstack([[0], quantile_diffs])

    # convert into quantiles
    # quantile_array1 = np.apply_along_axis(
    #     arr = sample_array1,
    #     axis = 0,
    #     func1d = lambda x: np.quantile(x[~np.isnan(x)], 
    #     q = quantile_values)
    # )

    # quantile_array2 = np.apply_along_axis(
    #     arr = sample_array2,
    #     axis = 0,
    #     func1d = lambda x: np.quantile(x[~np.isnan(x)], 
    #     q = quantile_values)
    # )
    if array1_quantile:
    	quantile_array1 = sample_array1
    else:
    	quantile_array1 = np.quantile(sample_array1[~np.isnan(sample_array1)], q = quantile_values)
    if array2_quantile:
    	quantile_array2 = sample_array2
    else:
    	quantile_array2 = np.quantile(sample_array2[~np.isnan(sample_array2)], q = quantile_values)

    # calculate distance between quantile
    dist = (np.absolute(quantile_array1 - quantile_array2)**p * quantile_diffs).sum()**(1/p)
    return dist

def pairwise_wasserstein(sample_array1, sample_array2, p, n_samples_min):
    '''
    description
    -----------
    calculate pairwise wasserstein distance between all the sampled distributions of one array and all the sampled distributions of another
    Wasserstein distance for dists U, V on real line: 
        int_{0}^1 (F^{-1}_U(q) - F^{-1}_V(q)) dq, where F^{-1}_A reps quantile function of rv A

    inputs
    ------
    sample_array1: N x S_max1 np array of floats/ints
        N is number of units in the array and S_max1 is the maximum number of samples across any of the N dists
    sample_array2: N x S_max2 np array of floats/ints
        N is number of units in the array and S_max2 is the maximum number of samples across any of the N dists
    p: order of wasserstein distance

    returns
    -------
    a N x N matrix where entry (i,j) represents Wasserstein-p distance between unit i and unit j
    '''

    # initialize return matrix s.t. entry (i,j) represents wasserstein-p distance 
        # between dist i of sample_array1 and dist j of sample_array2
    emd_matrix = np.ones((sample_array1.shape[0], sample_array2.shape[0]))
    for i in range(emd_matrix.shape[0]):
        for j in range(emd_matrix.shape[1]):
            # gather distributions i and j
            sample_array_i = sample_array1[i, :]
            sample_array_j = sample_array2[j, :]
            # calculate wasserstein-p distance
            emd_matrix[i, j] = wasserstein_dist(sample_array1 = sample_array_i,
                                                sample_array2 = sample_array_j, 
                                                p = p,
                                                n_samples_min = n_samples_min)

    return emd_matrix

def wasserstein2_barycenter(sample_array_1_through_n, weights, n_samples_min):
    '''
    description
    -----------
    compute the wasserstein-2 barycenter

    inputs
    ------
    sample_array_1_through_n : N x Smax numpy array with all of the samples from the distributional outcome
        N is number of units and Smax is the max number of samples from any distribution
        if some unit has S < Smax samples, then its entry of y should have (Smax - S) NA values
    weight : N x 1 array specifying the weight to place on each unit's distribution when taking average
    n_samples_min : minimum number of samples from any of the N distributions

    returns
    -------
    n_samples_min x 1 array such that entry i is (i/n_samples_min)-th quantile from barycenter
    '''

    # compute empirical quantile functions for each distribution 
    qtls_1_through_n = np.apply_along_axis(
                arr = sample_array_1_through_n,
                axis = 1,
                func1d = lambda x: np.quantile(
                    a = x, 
                    q = np.linspace(start = 0, stop = 1, num = n_samples_min)
                    )
                )

    # take quantile level euclidean average weighted by weights
    bcetner_qtl = np.average(a = qtls_1_through_n,
                             weights = weights, axis = 0)
#     bcetner_qtl = bcetner_qtl.reshape((bcetner_qtl.shape[0], 1))

    # return barycenter quantile function
    return bcetner_qtl

def calc_estimand(f_yi1, f_yi0, n_mc_samples):
	'''
	description
	-----------
	calculates P(Y_i1 > Y_i0 | Y_1 ~ f_yi1, Y_0 ~ f_yi0) via monte carlo sampling
	if f_yi is an empirical quantile function, we sample from the function
		-- akin to inverse transform sampling
	if f_yi is samples from pdf, we sample from the samples 
		-- akin to bootstrapping
	inputs
	------
	f_yi1 : S_i x 1 array of samples/quantiles from true treated potential distribution for unit i
	f_yi0 : S_i x 1 array of samples/quantiles from true control potential distribution for unit i
	n_mc_samples : number of monte carlo iterations to compute estimand

	returns
	-------
	unit i's individual treatment effect
	'''

	yi1_samples = np.random.choice(a = f_yi1, size = n_mc_samples, replace = True)
	yi0_samples = np.random.choice(a = f_yi0, size = n_mc_samples, replace = True)

	return (yi1_samples > yi0_samples).mean()


# class for pymaltspro 
class pymaltspro:
	def __init__(self, X, y, treatment, discrete = [], C = 1, k = 10, reweight = False):
		'''
		description
		-----------
		a class for running malts on distribution functions

		inputs
		------
		X : N x (p + 2) pandas dataframe with all of the input features and treatment variable
			N is number of units and p is number of input covariates
			 1 extra column for treatment variable
			 1 extra column for unit id s.t. row 1 is also unit 1
			make sure indices are in order 1, 2, ..., N so it aligns with y
		y : N x Smax numpy array with all of the samples from the distributional outcome
			N is number of units and Smax is the max number of samples from any distribution
			if some unit has S < Smax samples, then its entry of y should have (Smax - S) NA values
		treatment : string with name of treatment column
		# id_name : string with  name of id column
		discrete : list with names of discrete columns
		C : coefficient of regularization term
		k : number of neighbors to match with in caliper matching
		reweight : whether to reweight outcomes when combining
		'''
		# initialize data
		self.C = C
		self.k = k	
		self.reweight = reweight
		self.n, self.p = X.shape
		self.p = self.p - 1 # number of non-treatment input features
		self.treatment = treatment
		# self.id_name = id_name
		self.discrete = discrete
		self.continuous = list(set(X.columns).difference(set([treatment]+discrete)))
		# split data into control and treated units
		self.X_T = X.loc[X[treatment] == 1]
		self.X_C = X.loc[X[treatment] == 0]
		# split X dfs into discrete and continuous covariates
		self.Xc_T = self.X_T[self.continuous].to_numpy()
		self.Xc_C = self.X_C[self.continuous].to_numpy()
		self.Xd_T = self.X_T[self.discrete].to_numpy()
		self.Xd_C = self.X_C[self.discrete].to_numpy()
		self.y = y
		# N x Smax vectors Y
		# find the minimum number of samples taken from outcome dist across all units
		self.n_samples_min = np.apply_along_axis(
				arr = self.y,
				axis = 1,
				func1d = lambda x: x[x == x].shape[0]).min()
		self.quantile_values = np.linspace(start = 0, stop = 1, num = self.n_samples_min)
		self.quantile_diffs = self.quantile_values[1:] - self.quantile_values[:-1]

		self.Y_T = self.y[self.X_T.index.values]
		self.Y_C = self.y[self.X_C.index.values]
		self.Y_T_quantiles = np.apply_along_axis(
				arr = self.Y_T,
				axis = 1,
				func1d = lambda x: np.quantile(a = x[x == x], # remove NaN
								   q = self.quantile_values)
				)
		self.Y_C_quantiles = np.apply_along_axis(
				arr = self.Y_C,
				axis = 1,
				func1d = lambda x: np.quantile(a = x[x == x],  
								   q = self.quantile_values)
				)
		# store wasserstein distances in N_T x N_T (or N_C x N_C) matrix to avoid recomputing
		# self.Wass1_Y_T = pairwise_wasserstein(self.Y_T, self.Y_T, p = 1, n_samples_min = self.n_samples_min)
		# self.Wass1_Y_C = pairwise_wasserstein(self.Y_C, self.Y_C, p = 1, n_samples_min = self.n_samples_min)

		# pulled straight from pymalts2 code
		# Dc_T represents distance between continuous covariates for treatment units
		# Dc_C represents distance between continuous covariates for control units

		self.Dc_T = np.ones((self.Xc_T.shape[0],self.Xc_T.shape[1],self.Xc_T.shape[0])) * self.Xc_T.T
		# (X_{ik} - X^T_{jk})^2 --> compare kth cov between unit i and unit j
		self.Dc_T = (self.Dc_T - self.Dc_T.T) # each entry is a 3d matrix (i.e., cube) with entry (i,j,k) reps unit i, unit j, k-th covariate ==> check ordering of (i,j,k) 
		self.Dc_C = np.ones((self.Xc_C.shape[0],self.Xc_C.shape[1],self.Xc_C.shape[0])) * self.Xc_C.T # same thing for all controls
		self.Dc_C = (self.Dc_C - self.Dc_C.T) 

		# Dd_T represents distance between discrete covariates for treatment units
		# Dd_C represents distance between discrete covariates for control units
		self.Dd_T = np.ones((self.Xd_T.shape[0],self.Xd_T.shape[1],self.Xd_T.shape[0])) * self.Xd_T.T
		self.Dd_T = (self.Dd_T != self.Dd_T.T) 
		self.Dd_C = np.ones((self.Xd_C.shape[0],self.Xd_C.shape[1],self.Xd_C.shape[0])) * self.Xd_C.T
		self.Dd_C = (self.Dd_C != self.Dd_C.T) 

    # choose what kind of nearest neighbor we want; as of rn, it's just caliper (traditional knn)
	def threshold(self,x):
		'''
		description
		-----------
		chooses the k nearest neighbors between a given unit x and the rest in dataset

		input
		-----
		x : N_x x p array of covariates for the N_x units of interest

		returns
		-------
		N_x x k array with indexes of the k-nn for each unit
		'''
		# traditional knn; if we want to use exp(...), have to update this code: can take gradient of that
		k = self.k
		for i in range(x.shape[0]):
		    row = x[i,:]
		    row1 = np.where( row < row[np.argpartition(row,k+1)[k+1]],1,0)
		    x[i,:] = row1
		return x
    
    # calculates distance between two units _given_ a specified distance metric -- not being used right now
	def distance(self,Mc,Md,xc1,xd1,xc2,xd2):
		'''
		description
		-----------
		calculate the distance between two unit's covariates given a specified distance metric
		not being used currently
		'''
		dc = np.dot((Mc**2)*(xc1-xc2),(xc1-xc2))
		dd = np.sum((Md**2)*xd1!=xd2)
		return dc+dd

	def calcW_T(self,Mc,Md):
		'''
		description
		-----------
		weight matrix for each treated unit's outcome given the stretch (Mc, Md)
		gives the objective function to reweight the importance of units to a single imputation

		inputs
		------
		Mc : matrix of how to stretch each unit's continuous covariates
		Mc : matrix of how to stretch each unit's discrete covariates

		returns
		-------
		the weight matrix
		'''
	    #this step is slow
		Dc = np.sum( ( self.Dc_T * (Mc.reshape(-1,1)) )**2, axis=1)
		Dd = np.sum( ( self.Dd_T * (Md.reshape(-1,1)) )**2, axis=1)
		W = self.threshold( (Dc + Dd) )
		W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
		return W  

	def calcW_C(self,Mc,Md):
		'''
		description
		-----------
		return weight matrix for each control unit's outcome given the stretch (Mc, Md)
		gives the objective function to reweight the importance of units to a single imputation

		inputs
		------
		Mc : matrix of how to stretch each unit's continuous covariates
		Mc : matrix of how to stretch each unit's discrete covariates

		returns
		-------
		the weight matrix
		'''
	    #this step is slow
		Dc = np.sum( ( self.Dc_C * (Mc.reshape(-1,1)) )**2, axis=1)
		Dd = np.sum( ( self.Dd_C * (Md.reshape(-1,1)) )**2, axis=1)
		W = self.threshold( (Dc + Dd) )
		W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
		return W



	# combination of both W_C and W_T
	def Delta_(self,Mc,Md):
		'''
		description
		-----------
		calculate Delta for treated and control outcomes using `calcW_T` and `calcW_C`
			Delta_t = 1/N_t sum_{units i w/ treatment t} W1(f_Yi, wass2bary_i)
			wass2bary_i = argmin_{v_i} sum_{l : t_i = t_l} lambda_i W2(f_Yl, v_i)
			lambda_i = exp(-d_M(x_i, x_l))/[sum_{k : t_i = t_k} -d_M(x_i, x_k)]

		inputs
		------
		Mc : parameters of distance function such that dist btw cont covs a_c, b_c is
			d(a_c, b_c) = || Mc * a_c - Mc * b_c ||_2
		Md : parameters of distance function such that dist btw disc covs a_d, b_d is
			d(a_d, b_d) = sum_{discrete covs j}Md 1(a_{d,j} == b_{d,j})

		returns weighted or unweighted 
		'''
		self.W_T = self.calcW_T(Mc,Md)
		self.W_C = self.calcW_C(Mc,Md)
		self.delta_T = np.ones(shape = self.Y_T.shape[0])
		self.delta_C = np.ones(shape = self.Y_C.shape[0])

		for i in range(self.delta_T.shape[0]):
			Y_T_bary_i = wasserstein2_barycenter(
					sample_array_1_through_n = self.Y_T,
					weights = self.W_T[i, :],
					n_samples_min = self.n_samples_min
					)
			#W1(f_yi, wass2bary(distributions f_y1...f_yN_T, weights W_T[i]))
			self.delta_T[i] = wasserstein_dist(
				sample_array1 = self.Y_T[i, :],
				sample_array2 = Y_T_bary_i,
				p = 1, 
				n_samples_min = self.n_samples_min,
				array2_quantile = True
				)

		for i in range(self.delta_C.shape[0]):
			#W1(f_yi, wass2bary(distributions f_y1...f_yN_C, weights W_C[i]))
			Y_C_bary_i = wasserstein2_barycenter(
					sample_array_1_through_n = self.Y_C,
					weights = self.W_C[i, :],
					n_samples_min = self.n_samples_min
					)
			self.delta_C[i] = wasserstein_dist(
				sample_array1 = self.Y_C[i, :],
				sample_array2 = Y_C_bary_i,
				p = 1, 
				n_samples_min = self.n_samples_min,
				array2_quantile = True)

		self.delta_T = self.delta_T.mean()
		self.delta_C = self.delta_C.mean()
		
		# self.delta_T = np.sum((self.Y_T - (np.matmul(self.W_T,self.Y_T) - np.diag(self.W_T)*self.Y_T))**2)
		# self.delta_C = np.sum((self.Y_C - (np.matmul(self.W_C,self.Y_C) - np.diag(self.W_C)*self.Y_C))**2)
		if self.reweight == False:
		    return self.delta_T + self.delta_C
		elif self.reweight == True:
		    return (self.Y_T.shape[0] + self.Y_C.shape[0])*(self.delta_T/self.Y_T.shape[0] + self.delta_C/self.Y_C.shape[0])

	def objective(self, M):
		'''
		description
		-----------
		calculate objective function: min_M Delta_T + Delta_C + FrobeniusNorm(M)

		inputs
		------
		M : 1 x p vector specifying the weight to place on each cov
			discrete covariates come before continuous covariates

		returns
		-------
		calculated objective function
		'''
		Mc = M[ :len(self.continuous)]
		Md = M[len(self.continuous): ]
		delta = self.Delta_(Mc, Md)
		reg = self.C * ( np.linalg.norm(Mc,ord=2)**2 + np.linalg.norm(Md,ord=2)**2 )
		# ask harsh why we need cons1 and cons2
		cons1 = 0 * ( (np.sum(Mc) + np.sum(Md)) - self.p )**2
		cons2 = 1e+25 * np.sum( ( np.concatenate((Mc,Md)) < 0 ) )
		return delta + reg + cons1 + cons2

	# fits -- like sklearn fit
	def fit(self,method='COBYLA', M_init = None):
		'''
		description
		-----------
		find argument (i.e., param values) that minimize objective fn

		inputs
		------
		method: string specifying optimization method to use
		
		returns
		-------
		returns best fitting param values
		'''
		if M_init is None:
			M_init = np.ones((self.p,))
		res = opt.minimize( self.objective, x0=M_init,method=method )
		self.M = res.x
		self.Mc = self.M[:len(self.continuous)]
		self.Md = self.M[len(self.continuous):]
		self.M_opt = pd.DataFrame(self.M.reshape(1,-1),columns=self.continuous+self.discrete,index=['Diag'])
		return res

	# method for learning distance metric on train set
	# method for applying distance metric to test set
	# method for returning matched group observations for unit i
	# method for returning CATE given a matched group

	def get_matched_groups(self, X_estimation, Y_estimation, k = 10):
		'''
		description
		-----------
		create a dataframe that describes the matched group that each unit belongs to

		inputs
		------
		X_estimation : pandas dataframe of input features being used to estimate treatment effects
		Y_estimation : N_est by S_max_est numpy array s.t. i-th row has S_max_est entries
			Each entry of i-th row has all samples from unit i's outcome followed by NAs
		k : int describing number of nearest neighbors to match to

		returns
		-------
		returns tuple s.t. 
		first item is a pd dataframe of input features with index of matched units
		second item is the outcome ordered so that the i-th row of X df and Y align
		'''

		Xc = X_estimation[self.continuous].to_numpy()
		Xd = X_estimation[self.discrete].to_numpy()
		Y  = Y_estimation.to_numpy()
		T  = X_estimation[self.treatment].to_numpy()
		# splitted estimation data into treatment assignments for matching
		df_T = X_estimation.loc[X_estimation[self.treatment] == 1]
		df_C = X_estimation.loc[X_estimation[self.treatment] == 0]
		Y_T  = Y_estimation[df_T.index.values]
		Y_C  = Y_estimation[df_C.index.values]
		D_T = np.zeros((Y.shape[0],Y_T.shape[0]))
		D_C = np.zeros((Y.shape[0],Y_C.shape[0]))

		# distance of treated units
		Dc_T = (np.ones((Xc_T.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc_T.shape[0])) * Xc_T.T).T)
		Dc_T = np.sum( (Dc_T * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
		Dd_T = (np.ones((Xd_T.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd_T.shape[0])) * Xd_T.T).T )
		Dd_T = np.sum( (Dd_T * (self.Md.reshape(-1,1)) )**2 , axis=1 )
		D_T = (Dc_T + Dd_T).T

		# distance of control units
		Dc_C = (
			np.ones(
				(
		        	Xc_C.shape[0],
		        	Xc.shape[1],
		        	Xc.shape[0]
		    	)	
			) * Xc.T - 
				(
			np.ones(
				(
					Xc.shape[0],
					Xc.shape[1],
					Xc_C.shape[0])
				) * Xc_C.T).T
			)
		Dc_C = np.sum( (Dc_C * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
		Dd_C = (
			np.ones(
				(
					Xd_C.shape[0],
					Xd.shape[1],
					Xd.shape[0])
				) * Xd.T != (
			np.ones(
				(
					Xd.shape[0],
					Xd.shape[1],
					Xd_C.shape[0]
					)
				) * Xd_C.T).T )
		Dd_C = np.sum( (Dd_C * (self.Md.reshape(-1,1)) )**2 , axis=1 )
		D_C = (Dc_C + Dd_C).T

		MG = {}
		index = X_estimation.index
		for i in range(Y.shape[0]):
		    #finding k closest control units to unit i
		    idx = np.argpartition(D_C[i,:],k)
		    matched_df_C = pd.DataFrame( 
		    	np.hstack( 
		    		(
		    			Xc_C[idx[:k],:], 
		    			Xd_C[idx[:k],:].reshape((k,len(self.discrete))), 
		    			# Y_C[idx[:k]].reshape(-1,1), 
		    			D_C[i,idx[:k]].reshape(-1,1), 
		    			np.zeros((k,1)) ) 
		    		), 
		    	index = df_C.index[idx[:k]],
		    	columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment] 
		    	)

		    #finding k closest treated units to unit i
		    idx = np.argpartition(D_T[i,:],k)
		    matched_df_T = pd.DataFrame( 
		    	np.hstack( 
		    		(
		    			Xc_T[idx[:k],:], 
		    			Xd_T[idx[:k],:].reshape((k,len(self.discrete))), 
		    			# Y_T[idx[:k]].reshape(-1,1), 
		    			D_T[i,idx[:k]].reshape(-1,1), 
		    			np.ones((k,1)) ) 
		    		), 
		    	index=df_T.index[idx[:k]], 
		    	columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment] 
		    	)
		    matched_df = pd.DataFrame(
		    	np.hstack(
		    		(
		    			Xc[i], 
		    			Xd[i], 
		    			# Y[i], 
		    			0, 
		    			T[i])
		    		).reshape(1,-1), 
		    	index=['query'], 
		    	columns=self.continuous+self.discrete+[self.outcome,'distance',self.treatment]
		    	)
		    matched_df = matched_df.append(matched_df_T.append(matched_df_C))
		    MG[index[i]] = matched_df
		    
		MG_X_df = pd.concat(MG)
		# how to align input features to outcome variables? maybe need to store MG_Y in a dict?
		MG_Y_array = Y_estimation[MG_X_df.index]

# 

















