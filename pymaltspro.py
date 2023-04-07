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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings
# from scipy.statistics import wasserstein_distance as wd
warnings.filterwarnings("ignore")


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

def wasserstein_distance_unit_covs(x1, x2, weights, n_samples_min):
	distance = 0
	for j in range(x1.shape[1]):
		distance += weights[j] * wasserstein_dist(sample_array1=x1[j, :],
												sample_array2=x2[j, :],
												p = 1, 
												n_samples_min=n_samples_min, 
												array1_quantile=True, 
												array2_quantile=True)
	return distance

def wasserstein_distance_matrix(qtl_fn_matrix, weights):
	'''
	description
	-----------
	Calculate distance between all units' distributional covariates
	d(x_1, x_2) = \sum_{covs j} w_j W_1(x_{1,j}, x_{2,j}) 
	            = \sum_{covs j} w_j \int_0^1 |x_{1,j}(q) - x_{2,j}(q)| dq
		    	= \sum_{covs j} w_j \sum_{q = 0...1} |x_{1,j,q} - x_{2,j,q}|\Delta_q
	
	inputs
	------
	qtl_fn_matrix : N by P by Q matrix where
		N is number of units
		P is number of distributional covariates
		Q is number of discrete quantiles each covariate is evaluated at *with equal size bins*
		each entry is unit i's covariate j's empirical quantile q
	weights : P by 1 array where
		P is number of distributional covariates
	
	returns
	-------
	N by N matrix representing distributional distance between all units
	'''
	dist_matrix = np.zeros((qtl_fn_matrix.shape[0], qtl_fn_matrix.shape[0]))
	n_samples_min = qtl_fn_matrix.shape[2]
	for i1 in range(qtl_fn_matrix.shape[0]):
		for i2 in range(qtl_fn_matrix.shape[0]):
			dist_matrix[i1, i2] = wasserstein_distance_unit_covs(x1 = qtl_fn_matrix[i1, :, :],
																	x2 = qtl_fn_matrix[i2, :, :],
																	weights = weights,
																	n_samples_min= n_samples_min
																	)
	return dist_matrix



def wasserstein2_barycenter(sample_array_1_through_n, weights, n_samples_min, qtl_id):
	'''
	description
	-----------
	compute the wasserstein-2 barycenter

	inputs
	------
	sample_array_1_through_n : N x Smax numpy array with all of the samples from the distributional outcome
		N is number of units and Smax is the max number of samples from any distribution
		if some unit has S < Smax samples, then its entry of y should have (Smax - S) NA values
					OR
		N x Q matrix where Q is number of quantiles the quantile function is evaluated at
	weight : N x 1 array specifying the weight to place on each unit's distribution when taking average
	n_samples_min : minimum number of samples from any of the N distributions
	qtl_id : boolean set to True if sample_array_1_through_n is quantile function

	returns
	-------
	n_samples_min x 1 array such that entry i is (i/n_samples_min)-th quantile from barycenter
	'''
	# compute empirical quantile functions for each distribution 
	if qtl_id == False:
		qtls_1_through_n = np.apply_along_axis(
					arr = sample_array_1_through_n,
					axis = 1,
					func1d = lambda x: np.quantile(
						a = x, 
						q = np.linspace(start = 0, stop = 1, num = n_samples_min)
						)
					)
	else:
		qtls_1_through_n = sample_array_1_through_n

	# take quantile level euclidean average weighted by weights
	bcenter_qtl = np.average(a = qtls_1_through_n,
								weights = weights, axis = 0)

	# return barycenter quantile function
	return bcenter_qtl



def sample_quantile(quantile_fn, quantile):
    '''
    description
    -----------
    linearly interpolate quantile function and return value of a given quantile
    
    parameters
    ----------
    quantile_fn : numpy array with values of quantile function at specified quantiles
    quantile : value of quantile
    n_qtls : size of quantile function
    
    returns
    -------
    quantile function evaluated at specified quantile
    '''
    n_qtls = quantile_fn.shape[0] - 1
    quantile_index = quantile * n_qtls
    quantile_floor = int(np.floor(quantile_index))
    quantile_ceil  = int(np.ceil(quantile_index))
    if quantile_floor == quantile_ceil == quantile_index:
        return(quantile_fn[quantile_floor])
    else:
        return np.sum([quantile_fn[quantile_floor] * (quantile_index - quantile_floor), quantile_fn[quantile_ceil] * (quantile_ceil - quantile_index)])

def ITE(n_samples_min, y_true, y_impute, n_mc_samples, obs_treatment, y_true_qtl_id = False, y_impute_qtl_id = False):
    if y_true_qtl_id == False:
        y_true_qtl_fn = np.quantile(y_true, q = np.arange(n_samples_min)/n_samples_min)
    else:
        y_true_qtl_fn = y_true
    
    if y_impute_qtl_id == False:
        y_impute_qtl_fn = np.quantile(y_impute, q = np.arange(n_samples_min)/n_samples_min)
    else:
        y_impute_qtl_fn = y_impute
    
    qtls_to_sample = np.random.uniform(low = 0, high = 1, size = n_mc_samples)
    if obs_treatment == 1:
        y_treat = np.array([sample_quantile(quantile_fn = y_true_qtl_fn, quantile = q) for q in qtls_to_sample])
        y_control = np.array([sample_quantile(quantile_fn = y_impute_qtl_fn, quantile = q) for q in qtls_to_sample])
    else:
        y_treat = np.array([sample_quantile(quantile_fn = y_impute_qtl_fn, quantile = q) for q in qtls_to_sample])
        y_control = np.array([sample_quantile(quantile_fn = y_true_qtl_fn, quantile = q) for q in qtls_to_sample])
        
    return (y_treat > y_control).mean()

# class for pymaltspro 
class pymaltspro:
	def __init__(self, X, y, X_quantile_fns, treatment, discrete = [], C = 1, k = 10, reweight = False):
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
		self.p = self.p + X_quantile_fns.shape[1] - 1 # number of non-treatment input features
		self.treatment = treatment
		# self.id_name = id_name
		self.discrete = discrete
		self.continuous = list(set(X.columns).difference(set([treatment]+discrete)))
		self.distributional = range(X_quantile_fns.shape[1])
		# split data into control and treated units
		self.X_T = X.loc[X[treatment] == 1]
		self.X_C = X.loc[X[treatment] == 0]
		# split X dfs into discrete and continuous covariates
		self.Xc_T = self.X_T[self.continuous].to_numpy()
		self.Xc_C = self.X_C[self.continuous].to_numpy()
		self.Xd_T = self.X_T[self.discrete].to_numpy()
		self.Xd_C = self.X_C[self.discrete].to_numpy()

		# split covariates that are distributions into control/treated units
		self.Xq_T = X_quantile_fns[self.X_T.index.values, :, :]
		self.Xq_C = X_quantile_fns[self.X_C.index.values, :, :]

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

		# Dq_T represents distance between quantile functions for treatment units
		# Dq_C represents distance between quantile functions for control units
		self.Dq_T = wasserstein_distance_matrix(qtl_fn_matrix=self.Xq_T, weights = np.repeat(1, self.Xq_T.shape[1]))
		self.Dq_C = wasserstein_distance_matrix(qtl_fn_matrix=self.Xq_C, weights = np.repeat(1, self.Xq_T.shape[1]))

    # choose what kind of nearest neighbor we want; as of rn, it's just traditional knn
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
	def distance(self,Mc,Md, Mq, xc1,xd1,xc2,xd2, xq1, xq2):
		'''
		description
		-----------
		calculate the distance between two unit's covariates given a specified distance metric
		not being used currently
		'''
		dc = np.dot((Mc**2)*(xc1-xc2),(xc1-xc2))
		dd = np.sum((Md**2)*xd1!=xd2)
		dq = wasserstein_distance_unit_covs(x1 = xq1, 
				      						x2 = xq2,
											weights = Mq, 
											n_samples_min=xq1.shape[1])

		return dc+dd+dq

	def calcW_T(self,Mc,Md,Mq):
		'''
		description
		-----------
		weight matrix for each treated unit's outcome given the stretch (Mc, Md)
		gives the objective function to reweight the importance of units to a single imputation

		inputs
		------
		Mc : matrix of how to stretch each unit's continuous covariates
		Md : matrix of how to stretch each unit's discrete covariates
		Mq : matrix of how to stretch each unit's distributional covariates

		returns
		-------
		the weight matrix
		'''
	    #this step is slow
		Dc = np.sum( ( self.Dc_T * (Mc.reshape(-1,1)) )**2, axis=1)
		Dd = np.sum( ( self.Dd_T * (Md.reshape(-1,1)) )**2, axis=1)
		Dq = np.sum( ( self.Dq_T * (Mq.reshape(-1,1)) )**2, axis = 1 )
		W = self.threshold( (Dc + Dd + Dq) )
		W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
		return W  

	def calcW_C(self,Mc,Md,Mq):
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
		Dq = np.sum( ( self.Dq_T * (Mq.reshape(-1,1)) )**2, axis = 1 )
		W = self.threshold( (Dc + Dd) )
		W = W / (np.sum(W,axis=1)-np.diag(W)).reshape(-1,1)
		return W



	# combination of both W_C and W_T
	def Delta_(self,Mc,Md,Mq):
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
		self.W_T = self.calcW_T(Mc,Md,Mq)
		self.W_C = self.calcW_C(Mc,Md,Mq)
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
		Md = M[len(self.continuous):len(self.distributional)]
		Mq = M[len(self.distributional): ]
		delta = self.Delta_(Mc, Md, Mq)
		reg = self.C * ( np.linalg.norm(Mc,ord=2)**2 + np.linalg.norm(Md,ord=2)**2 + np.linalg.norm(Mq,ord=2))
		# ask harsh why we need cons1 and cons2
		cons1 = 0 * ( (np.sum(Mc) + np.sum(Md) + np.sum(Mq)) - self.p )**2
		cons2 = 1e+25 * np.sum( ( np.concatenate((Mc,Md,Mq)) < 0 ) )
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
		self.Md = self.M[len(self.continuous):len(self.distributional)]
		self.Mq = self.M[len(self.distributional):]
		self.M_opt = pd.DataFrame(self.M.reshape(1,-1),columns=self.continuous+self.discrete+self.distributional,index=['Diag'])
		return res

	def get_matched_groups(self, X_estimation, X_qtl_estimation, Y_estimation, k = 10):

		Xc = X_estimation[self.continuous].to_numpy()
		Xd = X_estimation[self.discrete].to_numpy()
		Y  = Y_estimation
		T  = X_estimation[self.treatment].to_numpy()
		# splitted estimation data into treatment assignments for matching
		df_T = X_estimation.loc[X_estimation[self.treatment] == 1]
		df_C = X_estimation.loc[X_estimation[self.treatment] == 0]
		Y_T  = Y_estimation[df_T.index.values, :]
		Y_C  = Y_estimation[df_C.index.values, :]
		D_T = np.zeros((Y.shape[0],Y_T.shape[0]))
		D_C = np.zeros((Y.shape[0],Y_C.shape[0]))

		# converting to numpy array
		Xc_T = df_T[self.continuous].to_numpy()
		Xc_C = df_C[self.continuous].to_numpy()
		Xd_T = df_T[self.discrete].to_numpy()
		Xd_C = df_C[self.discrete].to_numpy()
		Xq_T = X_qtl_estimation[df_T.index.values, :]
		Xq_C = X_qtl_estimation[df_C.index.values, :]

		# distance of treated units
		Dc_T = (np.ones((Xc_T.shape[0],Xc.shape[1],Xc.shape[0])) * Xc.T - (np.ones((Xc.shape[0],Xc.shape[1],Xc_T.shape[0])) * Xc_T.T).T)
		Dc_T = np.sum( (Dc_T * (self.Mc.reshape(-1,1)) )**2 , axis=1 )
		Dd_T = (np.ones((Xd_T.shape[0],Xd.shape[1],Xd.shape[0])) * Xd.T != (np.ones((Xd.shape[0],Xd.shape[1],Xd_T.shape[0])) * Xd_T.T).T )
		Dd_T = np.sum( (Dd_T * (self.Md.reshape(-1,1)) )**2 , axis=1 )
		Dq_T = wasserstein_distance_matrix(qtl_fn_matrix=Xq_T, weights = Mq)
		D_T = (Dc_T + Dd_T + Dq_T).T

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
		Dq_C = wasserstein_distance_matrix(qtl_fn_matrix=Xq_C, weights = Mq)
		D_C = (Dc_C + Dd_C + Dq_C).T

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
				columns=self.continuous+self.discrete+['distance',self.treatment] 
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
				columns=self.continuous+self.discrete+['distance',self.treatment] 
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
		#             index=['query'], 
				index = [i],
				columns=self.continuous+self.discrete+['distance',self.treatment]
				)
			matched_df = matched_df.append(matched_df_T.append(matched_df_C))
			matched_df['unit_treatment'] = X_estimation.loc[i, self.treatment]
			MG[index[i]] = matched_df
		#     return MG
		MG_X_df = pd.concat(MG).reset_index().rename(columns = {'level_0' : 'unit' ,  'level_1' : 'matched_unit'})

		return MG_X_df


	def barycenter_imputation(self, X_estimation, Y_estimation, MG, qtl_id):
	    Y_counterfactual = []
	    for i in X_estimation.index.values:
	        # make a holder list for adding matched units' outcomes
	        matched_unit_ids = MG.query('unit == ' + str(i)).query(self.treatment + ' != unit_treatment').matched_unit.values
	        matched_unit_outcomes = Y_estimation[matched_unit_ids, :]
	        y_i_counterfactual = wasserstein2_barycenter(
	            sample_array_1_through_n = matched_unit_outcomes, 
	            weights = np.repeat(1/matched_unit_outcomes.shape[0], matched_unit_outcomes.shape[0]),
	            n_samples_min=self.n_samples_min,
		    qtl_id=qtl_id
	        )
	        Y_counterfactual.append(y_i_counterfactual)
	    return np.array(Y_counterfactual)

	def sample_quantile(self, quantile_fn, quantile):
	    '''
	    description
	    -----------
	    linearly interpolate quantile function and return value of a given quantile
	    
	    parameters
	    ----------
	    quantile_fn : numpy array with values of quantile function at specified quantiles
	    quantile : value of quantile
	    n_qtls : size of quantile function
	    
	    returns
	    -------
	    quantile function evaluated at specified quantile
	    '''
	    n_qtls = quantile_fn.shape[0] - 1
	    quantile_index = quantile * n_qtls
	    quantile_floor = int(np.floor(quantile_index))
	    quantile_ceil  = int(np.ceil(quantile_index))
	#     return quantile_floor, quantile_ceil, quantile_index
	    if quantile_floor == quantile_ceil == quantile_index:
	        return(quantile_fn[quantile_floor])
	    else:
	        return np.sum([quantile_fn[quantile_floor] * (quantile_index - quantile_floor),
	            quantile_fn[quantile_ceil] * (quantile_ceil - quantile_index)])


	def ITE(self, y_estimation, y_cf, n_mc_samples, observed_treatment):
	    '''
	    description
	    -----------
	    Compute P(A > B | A ~ Y_i(1), B ~ Y_i(0))
	    
	    parameters
	    ----------
	    y_estimation : discrete quantiles for observed potential distribution in estimation set
	    y_cf : discrete quantiles for imputed potential distribution
	    n_mc_samples : number of monte carlo samples to use in evaluation
	    observed_treatment : the observed treatment assignment of unit i

	    returns
	    -------
	    a float representing probability of a treated sample being greater than untreated samples from potential distributions
	    '''
	    qtls_to_sample = np.random.uniform(0, 1, n_mc_samples)
	    if observed_treatment == 1:
	        y_treated = [self.sample_quantile(y_estimation, qtl) for qtl in qtls_to_sample]
	        y_control = [self.sample_quantile(y_cf, qtl) for qtl in qtls_to_sample]
	    else:
	        y_treated = [self.sample_quantile(y_cf, qtl) for qtl in qtls_to_sample]
	        y_control = [self.sample_quantile(y_estimation, qtl) for qtl in qtls_to_sample]
	    return (y_treated > y_control).mean()

	def linbo_ITE(self, y_obs, y_cf, observed_treatment, reference_distribution, y_obs_qtl_id = False):
	    '''
	    description
	    -----------
	    Compute Y_i^-1(1) \circ \lambda(t) - Y_i^{-1}(0) \circ \lambda(t)
	    
	    parameters
	    ----------
	    y_obs : array of samples/quantiles for the true observed outcome in estimation set
	    y_cf : array of quantile function for counterfactual outcome
	    observed_treatment : boolean that is True iff treated outcome observed, False otherwise
	    reference_distribution : a 2D array mapping samples from reference distribution to density of sample
	        -- reference distribution _must_ be continuous
	        -- col 1 is sample
	        -- col 2 is prob of observing sample
	    y_obs_qtl_id : boolean that is True iff y_estimation is a quantile function

	    returns
	    -------
	    E[Y_i(1)^{-1}(\lambda(t)) - Y_i(0)^{-1}(\lambda(t))], 0 <= t <= 1
	    '''
	    quantiles = np.linspace(start = 0, stop = 1, num = y_cf.shape[0])
	    if y_obs_qtl_id:
	        y_estimation = y_obs
	    else:
	        y_estimation = np.quantile(y_obs, quantiles)
	    
	    ylambda_treated = []
	    ylambda_control = []
	    if observed_treatment == 1:
	        for i in reference_distribution[1, :]:
	            ylambda_treated.append(self.sample_quantile(quantile_fn = y_estimation, quantile = i))
	            ylambda_control.append(self.sample_quantile(quantile_fn = y_cf, quantile = i))
	    else:
	        for i in reference_distribution[1, :]:
	            ylambda_treated.append(self.sample_quantile(quantile_fn = y_cf, quantile = i))
	            ylambda_control.append(self.sample_quantile(quantile_fn = y_estimation, quantile = i))

	    ylambda_treated = np.array(ylambda_treated)
	    ylambda_control = np.array(ylambda_control)
	    ylambda_treated_minus_control = ylambda_treated - ylambda_control
	    return_array = np.array([reference_distribution[0, :], ylambda_treated_minus_control])
	    return return_array
	
	def CATE(self, X_estimation, Y_estimation, reference_distribution, MG):
		CATE_array = []
		for i in X_estimation.index.values:
			# get treated units and outcomes that query unit is matched with
			
			treated_matched_unit_ids = MG.query('unit == ' + str(i)).query(self.treatment + ' == 1').matched_unit.values
			treated_matched_unit_outcomes = Y_estimation[treated_matched_unit_ids, :]
			# get quantile function of treated barycenter given X= x_i
			y_treated = wasserstein2_barycenter(
				sample_array_1_through_n = treated_matched_unit_outcomes, 
				weights = np.repeat(1/treated_matched_unit_outcomes.shape[0], treated_matched_unit_outcomes.shape[0]),
				n_samples_min=self.n_samples_min,
				qtl_id=False
			)
			
			# get control units and outcomes that query unit is matched with
			control_matched_unit_ids = MG.query('unit == ' + str(i)).query(self.treatment + ' == 0').matched_unit.values
			control_matched_unit_outcomes = Y_estimation[control_matched_unit_ids, :]
			# get quantile function control barycenter given X = x_i
			y_control = wasserstein2_barycenter(
				sample_array_1_through_n = control_matched_unit_outcomes, 
				weights = np.repeat(1/control_matched_unit_outcomes.shape[0], control_matched_unit_outcomes.shape[0]),
				n_samples_min=self.n_samples_min,
				qtl_id=False
			)
			# get Y^{-1} \circ \lambda for treated and control outcomes
			y_lambda_treated = []
			y_lambda_control = []
			for i in reference_distribution[1, :]:
				y_lambda_treated.append(self.sample_quantile(quantile_fn = y_treated, quantile = i))
				y_lambda_control.append(self.sample_quantile(quantile_fn = y_control, quantile = i))

			y_lambda_treated = np.array(y_lambda_treated)
			y_lambda_control = np.array(y_lambda_control)
			# estimate CATE and append to list
			CATE_i = y_lambda_treated - y_lambda_control

			CATE_array.append(CATE_i)
		CATE_array = np.array(CATE_array).reshape([Y_estimation.shape[0], reference_distribution[1, :].shape[0]])
		return CATE_array


	def mise(self, y_pred, y_true):
		'''
		description
		-----------
		Given function families u_i, v_i approximate 1/n sum_{i = 1}^n (int_{t} |u_i(t) - v_i(t)|^2 dt)

		parameters
		----------
		y_pred : array from predicted vector
		y_true : array from true vector

		returns
		-------
		float representing mean integrated squared error between two vectors
		'''
		return ((y_pred - y_true)**2).sum(axis = 1).mean()
	    















