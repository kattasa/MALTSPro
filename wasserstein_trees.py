#!/usr/bin/env python
# coding: utf-8

# # Methods and Code base for Wasserstein Decision Trees and Wasserstein Random Forests

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


def wasserstein2_barycenter(sample_array_1_through_n, weights, n_samples_min, quantile_id = False):
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
        quantile_id : boolean that is True if values in sample_array_1_through_n are empirical quantile functions
        returns
        -------
        n_samples_min x 1 array such that entry i is (i/n_samples_min)-th quantile from barycenter
        '''

        # compute empirical quantile functions for each distribution 
        if quantile_id:
            qtls_1_through_n = sample_array_1_through_n
        else:
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


# In[6]:


class wass_node:
    def __init__(self, X, y, y_quantile_id = False, min_samples_split=None, max_depth=None, depth=None, node_type=None, n_samples_min = None):
        # save data
        self.X = X.reset_index().drop('index', axis = 1)
        self.y = y
        self.features = X.columns
        
        self.y_quantile_id = y_quantile_id
        # find min number of samples in any y_i
        if n_samples_min is None:
            self.n_samples_min = np.apply_along_axis(
                arr = self.y,
                axis = 1,
                func1d = lambda x: x[x == x].shape[0]).min()
        else:
            self.n_samples_min = n_samples_min
        
        # save hyperparameters: defaults are 20 and 5 unless specified otherwise
        if min_samples_split is None:
            self.min_samples_split = 20
        else:
            self.min_samples_split = min_samples_split
        if max_depth is None:
            self.max_depth = 5
        else:
            self.max_depth = max_depth
#         self.max_depth = self.max_depth if max_depth is not None else 5
        
        # current node characteristics: defaults are 0 depth and 'root' node
        if depth is None:
            self.depth = 0
        else:
            self.depth = depth
        if node_type is None:
            self.node_type = 'root'
        else:
            self.node_type = node_type
        
        # number of units in the node
        self.n_units = X.shape[0]
        
        # get barycenter of current node
        self.barycenter = wasserstein2_barycenter(sample_array_1_through_n = y,
                                                  quantile_id=y_quantile_id,  
                                                  n_samples_min = self.n_samples_min, 
                                                  weights = np.repeat(a = 1/self.n_units, repeats = self.n_units)
                                                 )
        
        # calculate mean squared wasserstein error: 1/N sum_i W_2(bary, y_i)^2
        self.mswe = self.self_mswe_calc()
        
        # initialize feature/value to split on
        self.best_feature = None
        self.best_feature_split_val = None
        
        # initializing child nodes
        self.left_node = None
        self.right_node = None
        
    def self_mswe_calc(self):
        mswe = np.apply_along_axis(arr = self.y, 
            axis = 1, 
            func1d = lambda x: wasserstein_dist(sample_array1 = x, 
                                                sample_array2 = self.barycenter, 
                                                p = 2, 
                                                n_samples_min = self.n_samples_min, 
                                                array1_quantile = self.y_quantile_id,
                                                array2_quantile = True
                                               ) ** 2
           ).mean()
        return mswe

        
        
    def _mswe(self, y_true, y_pred, y_true_quantile_id = False, y_pred_quantile_id = True):
        squared_wass_dist_array = np.apply_along_axis(
            arr = y_true,
            axis = 1,
            func1d = lambda x: wasserstein_dist(sample_array1 = x,
                                                sample_array2 = y_pred, 
                                                p = 2,
                                                n_samples_min = self.n_samples_min, 
                                                array1_quantile = y_true_quantile_id,
                                                array2_quantile = y_pred_quantile_id, 
                                               ) ** 2
        )
        mswe = squared_wass_dist_array.mean()
        return mswe
    
    def best_split(self):
        
        mswe_base = self.mswe
        best_feature = None
        best_feature_split_val = None
        best_mswe = np.inf
        
        for feature in self.features:
            # split each feature at its median
            feature_split_val = self.X[feature].mean()
            
            # left node would be whenever x_feature <= median
            X_left = self.X.loc[self.X[feature] <= feature_split_val]
            X_right = self.X.loc[self.X[feature] > feature_split_val]
            
            y_left = self.y[X_left.index]
            y_right = self.y[X_right.index]
            
            if X_left.shape[0] > 0 and X_right.shape[0] > 0:
                # calculate barycenter of left and right nodes
                y_left_bary = wasserstein2_barycenter(sample_array_1_through_n = y_left,
                                                      weights = np.repeat(a = 1/X_left.shape[0], repeats = X_left.shape[0]), 
                                                      n_samples_min = self.n_samples_min,
                                                      quantile_id = self.y_quantile_id
                                                     )
                # calculate barycenter of left and right nodes
                y_right_bary = wasserstein2_barycenter(sample_array_1_through_n = y_right,
                                                      weights = np.repeat(a = 1/X_right.shape[0], repeats = X_right.shape[0]), 
                                                      n_samples_min = self.n_samples_min,
                                                      quantile_id = self.y_quantile_id
                                                     )
                # calculate mswe for left and right nodes
#                 left_mswe = self._mswe(y_true = y_left,
#                                        y_pred = y_left_bary,
#                                        y_true_quantile_id = self.y_quantile_id,
#                                        y_pred_quantile_id = True)
                left_mswe = np.apply_along_axis(
                    arr = y_left, 
                    axis = 1, 
                    func1d = lambda x: wasserstein_dist(sample_array1 = x, 
                                                        sample_array2 = y_left_bary, 
                                                        p = 2, 
                                                        n_samples_min = self.n_samples_min, 
                                                        array1_quantile = self.y_quantile_id,
                                                        array2_quantile = True
                                                       ) ** 2
                   ).mean()
                # calculate mswe for left and right nodes
                right_mswe = np.apply_along_axis(
                    arr = y_right, 
                    axis = 1, 
                    func1d = lambda x: wasserstein_dist(sample_array1 = x, 
                                                        sample_array2 = y_right_bary, 
                                                        p = 2, 
                                                        n_samples_min = self.n_samples_min, 
                                                        array1_quantile = self.y_quantile_id,
                                                        array2_quantile = True
                                                       ) ** 2
                   ).mean()

                # average both mswe together
                total_mswe = ((X_left.shape[0]) * left_mswe + (X_right.shape[0]) * right_mswe)/self.X.shape[0]

                # check if we improved mswe
                if total_mswe < best_mswe:
                    best_feature = feature
                    best_feature_split_val = feature_split_val
                    best_mswe = total_mswe
#                 print(feature, total_mswe)
        return (best_feature, best_feature_split_val)
    
    def grow_tree(self):
        if (self.depth < self.max_depth) and (self.n_units > self.min_samples_split):
            best_feature, best_feature_split_val = self.best_split()
            if best_feature is not None:
                self.best_feature = best_feature
                self.best_feature_split_val = best_feature_split_val
                
                X_left = self.X.loc[self.X[best_feature] <= best_feature_split_val]
                X_right = self.X.loc[self.X[best_feature] > best_feature_split_val]

                y_left = self.y[X_left.index]
                y_right = self.y[X_right.index]

                left = wass_node(
                    X = X_left,
                    y = y_left,
                    y_quantile_id = self.y_quantile_id,
                    min_samples_split = self.min_samples_split,
                    max_depth = self.max_depth,
                    depth = self.depth + 1,
                    node_type = 'left_node'
                )

                if left is not None:
                    self.left_node = left
                    try:
                        self.left_node.grow_tree()
                    except:
                        print(self.left_node.X.shape)

                right = wass_node(
                    X = X_right,
                    y = y_right,
                    y_quantile_id = self.y_quantile_id,
                    min_samples_split = self.min_samples_split,
                    max_depth = self.max_depth,
                    depth = self.depth + 1,
                    node_type = 'right_node'
                )

                if right is not None:
                    self.right_node = right
                    self.right_node.grow_tree()
                
    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        elif self.node_type == 'left_node':
            if self.best_feature is not None:
                print(f"|{spaces} Split rule: {self.best_feature} <= {self.best_feature_split_val}")
        else:
            if self.best_feature is not None:
                print(f"|{spaces} Split rule: {self.best_feature} > {self.best_feature_split_val}")
        print(f"{' ' * const}   | MSWE of the node: {round(self.mswe, 5)}")
        print(f"{' ' * const}   | Count of observations in node: {self.n_units}")
#         print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")   

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left_node is not None: 
            self.left_node.print_tree()
        
        if self.right_node is not None:
            self.right_node.print_tree()
              
    def predict(self, X_valid):
        node = self
        y_pred = []
        for col in self.X.columns:
            if col not in X_valid.columns:
              raise Exception(f'{col} is not a valid column')
        for i in X_valid.index.values:
            while (node.left_node is not None) and (node.right_node is not None):
                if X_valid.loc[i, node.best_feature] <= node.best_feature_split_val:
                    node = node.left_node
                else:
                    node = node.right_node
            y_pred_i = wasserstein2_barycenter(sample_array_1_through_n = node.y,
                    weights = np.repeat(1/node.y.shape[0], node.y.shape[0]),
                    n_samples_min = node.n_samples_min, 
                    quantile_id = node.y_quantile_id)
            y_pred.append(y_pred_i)
        return y_pred
              
#     def predict(self, X_valid):
#         for i in X_valid.


# In[8]:


class wass_forest:
    def __init__(self, X, y, y_quantile_id = False, min_samples_split=None, max_depth=None, depth=None, node_type=None, n_trees = 20, seed = 999, n_samples_min = None):
        self.X = X
        self.y = y
        self.y_quantile_id = y_quantile_id
        
        # find min number of samples in any y_i
        if n_samples_min is None:
            self.n_samples_min = np.apply_along_axis(
                arr = self.y,
                axis = 1,
                func1d = lambda x: x[x == x].shape[0]).min()
        else:
            self.n_samples_min = n_samples_min
        
        # save hyperparameters: defaults are 20 and 5 unless specified otherwise
        if min_samples_split is None:
            self.min_samples_split = 20
        else:
            self.min_samples_split = min_samples_split
        if max_depth is None:
            self.max_depth = 5
        else:
            self.max_depth = max_depth
#         self.max_depth = self.max_depth if max_depth is not None else 5
        
        # current node characteristics: defaults are 0 depth and 'root' node
        if depth is None:
            self.depth = 0
        else:
            self.depth = depth
        if node_type is None:
            self.node_type = 'root'
        else:
            self.node_type = node_type
            
        self.trees = []
        
        # for each new tree
        for i in range(n_trees):
            # bootstrap data: choose index
            bootstrap_ids = np.random.choice(a = range(self.X.shape[0]), size = self.X.shape[0], replace = True)
            bootstrap_X = []
            bootstrap_y = []
            for index in bootstrap_ids:
                bootstrap_X.append(pd.DataFrame(self.X.iloc[index, :]).transpose())
                bootstrap_y.append(self.y[index, :])
            bootstrap_X_df = pd.concat(bootstrap_X, axis = 0)
            bootstrap_y_np = np.array(bootstrap_y)
            
            # grow tree
            wass_tree = wass_node(
                X = bootstrap_X_df,
                y = bootstrap_y_np,
                y_quantile_id = self.y_quantile_id,
                min_samples_split = self.min_samples_split,
                max_depth = self.max_depth, 
                n_samples_min = self.n_samples_min
            )
            
            wass_tree.grow_tree()
            
            # save tree
            self.trees.append(wass_tree)
        
    def predict(self, X_valid):
        # initialize empty list to store predictions from individual trees
        y_pred = []
        # get each prediction for each tree
        for wass_tree in self.trees:
            y_pred.append(wass_tree.predict(X_valid))
        # compute the barycenter
        y_pred_np = np.array(y_pred) # should be a T by p matrix, where T is #trees
        y_bary = wasserstein2_barycenter(
            sample_array_1_through_n = y_pred_np,
            weights = np.repeat(1/len(self.trees), len(self.trees)),
            n_samples_min = self.n_samples_min,
            quantile_id = True)
        return y_bary
    
    


# # Simulations

# In[5]:


# np.random.seed(99)
# n_units = 1000
# n_samples_per_unit = 1001
# y_values = []
# x1_values = []
# x2_values = []
# x3_values = [] # nuisance parameter
# treated_values = []
# for i in range(n_units):
#     beta_mean = -1
#     x1 = np.random.binomial(n = 1, p = 0.7, size = 1)
#     if x1 == 0:
#         x2 = np.random.binomial(n = 1, p = 0.5, size = 1)
#         if x2 == 0:
#             treated = np.random.binomial(n = 1, p = 0.6, size = 1)
#             if treated == 0:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 2, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 40, size = n_samples_per_unit)
#             if treated == 1:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 4, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 80, size = n_samples_per_unit)            
#         if x2 == 1:
#             treated = np.random.binomial(n = 1, p = 0.5, size = 1)
#             if treated == 0:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 1.5, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 30, size = n_samples_per_unit)
#             if treated == 0:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 4.5, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 90, size = n_samples_per_unit)
#     if x1 == 1:
#         x2 = np.random.binomial(n = 1, p = 0.4, size = 1)
#         if x2 == 0:
#             treated = np.random.binomial(n = 1, p = 0.4, size = 1)
#             if treated == 0:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 2.5, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 50, size = n_samples_per_unit)
#             if treated == 1:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 5, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 100, size = n_samples_per_unit)
#         if x2 == 1:
#             treated = np.random.binomial(n = 1, p = 0.3, size = 1)
#             if treated == 0:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 1, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 20, size = n_samples_per_unit)
#             if treated == 1:
#                 while beta_mean <= 0:
#                     beta_mean = np.random.normal(loc = 6, scale = 0.05, size = 1)
#                 y_i = np.random.beta(a = beta_mean, b = 120, size = n_samples_per_unit)
#     x3 = np.random.binomial(n = 1, p = 0.5, size = 1)
#     y_values.append(y_i)
#     x1_values.append(x1[0])
#     x2_values.append(x2[0])
#     x3_values.append(x3[0])
#     treated_values.append(treated[0])

# Y = np.array(y_values)
# X_df = pd.DataFrame(
#     {
#         'x1' : x1_values,
#         'x2' : x2_values,
#         'x3' : x3_values,
#         'treatment' : treated_values,
#     }
# )

# np.random.seed(888)
# train_id = np.random.choice(a = X_df.index.values, size = int(0.8 * X_df.shape[0]), replace = False)

# X_train = X_df.iloc[train_id]
# y_train = Y[train_id]

# X_valid = X_df.drop(train_id, axis = 0)
# y_valid = Y[X_valid.index]

# wass_tree = wass_node(
#                 X = X_df.drop('x3', axis = 1),
#                 y = Y,
#                 y_quantile_id = False,
#                 min_samples_split = 20,
#                 max_depth = 4
#             )

# wass_tree.grow_tree()

# # # # wass_tree.predict(X_valid)
# # def predict(node, X_valid):
# # #     node = self
# #     y_pred = []
# #     for col in node.X.columns:
# #         if col not in X_valid.columns:
# #             raise Exception('{col} is not a valid column')
# #     for i in range(X_valid.shape[0]):
# #         while (node.left_node is not None) and (node.right_node is not None):
# #             if X_valid.loc[i, node.best_feature] <= node.best_feature_split_val:
# #                 node = node.left_node
# #             else:
# #                 node = node.right_node
# #         y_pred.append(y_pred)
# #     return y_pred

# # predict(wass_tree, X_valid)
# node = wass_tree
# y_pred = []
# for col in node.X.columns:
#     if col not in X_valid.columns:
#         raise Exception('{col} is not a valid column')
# for i in X_valid.index:
#     while (node.left_node is not None) and (node.right_node is not None):
#         if X_valid.loc[i, node.best_feature] <= node.best_feature_split_val:
#             node = node.left_node
#         else:
#             node = node.right_node
#     y_pred_i = wasserstein2_barycenter(sample_array_1_through_n = node.y, 
#                                        weights = np.ones(node.y.shape[0])/node.y.shape[0], 
#                                        n_samples_min = node.n_samples_min, 
#                                        quantile_id = False
#                                       )
#     y_pred.append(y_pred_i)
# # return y_pred

# def mswe(y_true, y_pred, n_samples_min, y_true_quantile_id = False, y_pred_quantile_id = True):
#     squared_wass_dist_array = np.apply_along_axis(
#         arr = y_true,
#         axis = 1,
#         func1d = lambda x: wasserstein_dist(sample_array1 = x,
#                                             sample_array2 = y_pred, 
#                                             p = 2,
#                                             n_samples_min = n_samples_min, 
#                                             array1_quantile = y_true_quantile_id,
#                                             array2_quantile = y_pred_quantile_id, 
#                                            ) ** 2
#     )
#     mswe = squared_wass_dist_array.mean()
#     return mswe

# mswe(y_true = y_valid, 
#      y_pred = np.array(y_pred), 
#      y_true_quantile_id = False,
#      y_pred_quantile_id = True,
#      n_samples_min = 1001
#     )

