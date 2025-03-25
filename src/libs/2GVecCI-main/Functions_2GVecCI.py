import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as plt
import sklearn
from sklearn import linear_model
import copy
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.cmiknn import CMIknn
import time
import itertools
import os
import ast
from sklearn.gaussian_process import GaussianProcessRegressor as GPReg
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from hyppo.independence import Dcorr

import lingam


##FUNCTIONS FOR TEST DATA GENERATION

def sparse_randomcoeffmatrix(shape = (1,1), sparsity = 1.0, random_state = None, data_rvs = None):
    ##generates a sparse random matrix, random state refers to the random state that chooses the nonzero entry,
    ## data_rvs specifies the method of the random state that chooses the entries.By default it is random_state.uniform
    return sp.sparse.random(shape[0],shape[1],density=sparsity, random_state=random_state, data_rvs = data_rvs).A

def generate_random_contemp_model(N, L, random_state=None):
    ##Generates a random DAG over N scalar variables with L links. The DAG is encoded as in the software package tigramite to ensure
    ##compatibility

    if random_state is None:
        random_state = np.random

    # Random order
    causal_order = list(random_state.permutation(N))
    links = dict([(i, ([], [])) for i in range(N)])

    # Create contemporaneous DAG
    chosen_links = []
    contemp_links = []
    for l in range(L):
        effect = random_state.choice(causal_order[1:])
        cause = random_state.choice(causal_order[:causal_order.index(effect)])
        while (cause, effect) in chosen_links:
            effect = random_state.choice(causal_order[1:])
            cause = random_state.choice(causal_order[:causal_order.index(effect)])
        contemp_links.append((cause, effect))
        chosen_links.append((cause, effect))

    for (i, j) in contemp_links:
        strength = random_state.uniform(0.1, 1.)
        links[j][0].append(int(i))
        links[j][1].append(strength)
    return links

def generate_random_diagonal_matrix(low=0.2, high=2.0, size=1, random_state =None):
    ##generates a random diagonal matrix of shape size x size with uniformly random diagonal entries between low and high.
    if random_state is None:
        random_state = np.random
    return np.diag(random_state.uniform(low,high,size))

def normalized_weibull(random_state, samplesize = 1000):
    ##generates samples from a normalized weibull distribution that can be used as non-Gaussian noise.
    a=2
    mean = sp.special.gamma(1. / a + 1)
    variance = sp.special.gamma(2. / a + 1) - sp.special.gamma(1. / a + 1) ** 2
    return (random_state.weibull(a=a, size=samplesize) - mean)/np.sqrt(variance)

def internal_linear_scm(links, samplesize=1000, random_state=None, noise_type = 'gaussian', low = 0.2, high = 2.0):
    ##generates a random linear SCM over a prescribed DAG.
    ##INPUT: List of links of the DAG as outputted by generate_random_contemp_model,
    ##see the documentation of Tigramite for details on the lists structure.
    ## noise_type = string, either 'gaussian' or 'mixed' for either Gaussian or Gaussian and Weibull noises
    ##OUTPUT: dataframe containing samplesize samples for each variable in the DAG.
    if random_state is None:
        random_state = np.random
    variables = links.keys()
    cov = generate_random_diagonal_matrix(low=low, high=high, size=len(variables))
    if noise_type == 'gaussian':
        mean = np.zeros(len(variables))
        noise = random_state.multivariate_normal(mean, cov, size=samplesize)
    elif noise_type == 'mixed':
        noise_dists = ['gaussian', 'weibull']
        noise = np.zeros(shape=(samplesize,len(variables)))
        for var in variables:
            noise_dist = random_state.choice(noise_dists)
            variance = cov[var,var]
            sigma = np.sqrt(variance)
            if noise_dist == 'weibull':
                weib = normalized_weibull(random_state,samplesize)
                for j in range(samplesize):
                    noise[j, var] = sigma*weib[j]
            if noise_dist == 'gaussian':
                gau = random_state.normal(size=samplesize)
                for j in range(samplesize):
                    noise[j, var] = sigma*gau[j]
    else:
        raise ValueError("Unknown noise type")

    for var in variables:

        for cause in links[var][0]:
            for j in range(samplesize):
                noise[j, var] += noise[j, cause] * links[var][1][links[var][0].index(cause)]

    return pp.DataFrame(noise, var_names=variables)


def standardize_DataFrame(ppdataframe):
##function for standardization of data. Standardization does not affect performance as can be checked empirically using this function.
    size = len(ppdataframe.var_names)
    array = ppdataframe.values[0]
    new_array = np.zeros(shape= array.shape)
    for i in range(size):
        var_mean = array[:,i].mean()
        var_std =  array[:,i].std()
        if var_std != 0.0:
            new_array[:,i] = (array[:,i] - var_mean)/var_std
    return pp.DataFrame(new_array, var_names=ppdataframe.var_names)

class biv_model:
    ##biv_model class that has methods for generating cause and effect regions with prescribed link numbers and prescribed
    ##interaction sparsity. The effect region can depend linearly on the cause region or linearly on the squares of the cause
    ##region
    def __init__(self, cause_no=1, effect_no=1, cause_link_no=0, effect_link_no=0, sparsity=1.0, samplesize=1000,
                 random_state=None, noise_type = 'gaussian'):
        self.cause_no = cause_no
        self.effect_no = effect_no
        self.cause_link_no = cause_link_no
        self.effect_link_no = effect_link_no
        self.sparsity = sparsity
        self.samplesize = samplesize
        self.state = random_state
        self.noise_type = noise_type

    def build_cause_region(self):
        self.cause_links = generate_random_contemp_model(self.cause_no, self.cause_link_no, random_state=self.state)
        self.cause = internal_linear_scm(self.cause_links, self.samplesize, self.state, noise_type=self.noise_type)

    def build_interaction_matrix(self):
        self.interaction_matrix = sparse_randomcoeffmatrix((self.effect_no, self.cause_no), self.sparsity)

    def build_effect_region(self):
        self.effect_links = generate_random_contemp_model(self.effect_no, self.effect_link_no, random_state=self.state)
        self.effect_internal = internal_linear_scm(self.effect_links, self.samplesize, self.state, noise_type=self.noise_type)
        Y_vector = self.effect_internal.values[0]
        for i in range(self.samplesize):
            Y_vector[i][:] += np.matmul(self.interaction_matrix, self.cause.values[0][i][:])
        self.effect = pp.DataFrame(Y_vector)

    def build_quadratic_effect_region(self):
        self.effect_links = generate_random_contemp_model(self.effect_no, self.effect_link_no, random_state=self.state)
        self.effect_internal = internal_linear_scm(self.effect_links, self.samplesize, self.state, noise_type=self.noise_type)
        Y_vector = self.effect_internal.values
        for i in range(self.samplesize):
            Y_vector[i][:] += np.matmul(self.interaction_matrix, self.cause.values[i][:]*self.cause.values[i][:])
        self.effect = pp.DataFrame(Y_vector)

    def build_model(self):
        self.build_cause_region()
        self.build_interaction_matrix()
        self.build_effect_region()

    def build_quadratic_interaction_model(self):
        self.build_cause_region()
        self.build_interaction_matrix()
        self.build_quadratic_effect_region()


##FUNCTIONS FOR CAUSAL INFERENCE

def regression(X,Y,fit_intercept=False):
    ##Regression function that regresses the random vector X linearly on Y and Y on X and returns both residuals
    ##INPUT: two numpy arrays X, Y of dimension samplesize x vector length
    ##OUTPUT dictionary containing the residuals of the regression of Y on X (key 'X_to_Y') and X on Y (key 'Y_to_X').
    dict = {}
    reg_X_to_Y = linear_model.LinearRegression(fit_intercept=False)
    reg_Y_to_X = linear_model.LinearRegression(fit_intercept=False)
    reg_X_to_Y.fit(X.values[0],Y.values[0])
    reg_Y_to_X.fit(Y.values[0],X.values[0])
    reg_X_to_Y.residuals = pp.DataFrame((Y.values[0] - reg_X_to_Y.predict(X.values[0])), var_names=Y.var_names)
    reg_Y_to_X.residuals = pp.DataFrame((X.values[0] - reg_Y_to_X.predict(Y.values[0])), var_names=X.var_names)
    dict['X_to_Y'] = reg_X_to_Y
    dict['Y_to_X'] = reg_Y_to_X
    return dict

def GP_regression(X,Y,random_state = None):
    ##Regression function that regresses the random vector X on Y and Y on X with a Gaussian process regression
    ## and returns both residuals
    ##INPUT: two numpy arrays X, Y of dimension samplesize x vector length
    ##OUTPUT dictionary containing the residuals of the regression of Y on X (key 'X_to_Y') and X on Y (key 'Y_to_X').
    if random_state is None:
        random_state = np.random
    dict = {}
    kernel = Matern() + ConstantKernel()
    reg_X_to_Y = GPReg(kernel=kernel,random_state=random_state)
    reg_Y_to_X = GPReg(kernel=kernel,random_state=random_state)
    reg_X_to_Y.fit(X.values[0], Y.values[0])
    reg_Y_to_X.fit(Y.values[0], X.values[0])
    reg_X_to_Y.residuals = pp.DataFrame((Y.values[0] - reg_X_to_Y.predict(X.values[0])), var_names=Y.var_names)
    reg_Y_to_X.residuals = pp.DataFrame((X.values[0] - reg_Y_to_X.predict(Y.values[0])), var_names=X.var_names)
    dict['X_to_Y'] = reg_X_to_Y
    dict['Y_to_X'] = reg_Y_to_X
    return dict


def RFDC(X,Y, Z=np.array([]), random_state=None, n_estimators=100, n_jobs = -1, max_samples = None, Dcorr_reps = 1000):
    #RandomForest Distance Correlation Conditional Independence test
    if random_state == None:
        random_state = np.random
    if Z.size==0:
        val, pval = Dcorr().test(X, Y, reps=Dcorr_reps, workers=-1)
    else:
        reg_X_on_Z = RandomForestRegressor(n_estimators, n_jobs=n_jobs, random_state=random_state, max_samples=max_samples)
        reg_Y_on_Z = RandomForestRegressor(n_estimators, n_jobs=n_jobs, random_state=random_state, max_samples=max_samples)
        reg_X_on_Z.fit(Z, X)
        reg_Y_on_Z.fit(Z, Y)
        reg_X_on_Z.residuals = X - reg_X_on_Z.predict(Z)
        reg_Y_on_Z.residuals = Y - reg_Y_on_Z.predict(Z)
        val, pval = Dcorr().test(x=reg_X_on_Z.residuals, y=reg_Y_on_Z.residuals, reps= Dcorr_reps, workers=-1)
    return val, pval




def full_conditioning_ind_test(X, Y, alpha=0.01, type='both', CI_test_method='ParCorr', linear = True, fit_intercept = False, random_state = None):
    '''
    Implementation of 2G-VecCI.Full as desribed in the submitted article.
    Runs sparsity based independent test of regions X and Y with the prescribed CI_test_method.
    
    Parameters:
        X: pp.Dataframes containing regional variables,
        alpha: floating number, significance level for conditional independence testing
        type: string, either 'both' or 'minimal'. Conditional Independences are tested either on both
            regions or on the smaller region
        CI_test_method: The conditional independence test. Options: 'ParCorr', 'GPDC', 'CMIknn', see the documentation
            of Tigramite for details on the implementations of these tests
        linear: string, Options: 'yes' or 'no'. If yes, conditioning on regions is handled by an OLS regression step and further
            tests are run on the residuals . If no, X is added to all conditioning sets when testing for Y|X densities and vice versa.
    
    Returns:
    dictionary describing number of detected edges on regions and their residuals as well as sparsity,
    measured as number of detected edges/number of edges on fully connected graph
    '''
    
    if random_state == None:
        random_state = np.random

    if CI_test_method == 'RFDC':
        dict = {}
        if type == 'both':
            edgecounterY = 0
            edgecounterResY = 0
            max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
            max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
            for var1 in range(len(Y.var_names)):
                for var2 in range(var1 + 1, len(Y.var_names)):
                    removedY = np.delete(Y.values[0], (var1, var2), 1)
                    valY, pvalY = RFDC(np.array(Y.values[0][:, var1:var1 + 1]), np.array(Y.values[0][:, var2:var2 + 1]),
                                                       Z=removedY, random_state=random_state)
                    if pvalY < alpha:
                        edgecounterY += 1
                    cond_setY = np.concatenate((removedY, X.values[0]), axis=1)
                    valResY, pvalResY = RFDC(np.array(Y.values[0][:, var1:var1 + 1]), np.array(Y.values[0][:, var2:var2 + 1]),
                                                             Z=cond_setY,random_state=random_state)
                    if pvalResY < alpha:
                        edgecounterResY += 1
                    print(pvalY, pvalResY)
            dict['number of nodes Y'] = len(Y.var_names)
            dict[' max number of edges Y'] = max_edgenumberY
            dict['number of edges Y'] = edgecounterY
            dict['number of edges Res Y'] = edgecounterResY
            dict['edge density of Y'] = edgecounterY / max_edgenumberY
            dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
            edgecounterX = 0
            edgecounterResX = 0
            for var1 in range(len(X.var_names)):
                for var2 in range(var1 + 1, len(X.var_names)):
                    removedX = np.delete(X.values[0], (var1, var2), 1)
                    valX, pvalX = RFDC(np.array(X.values[0][:, var1:var1 + 1]), np.array(X.values[0][:, var2:var2 + 1]),
                                                       Z=np.array(removedX),random_state=random_state)
                    if pvalX < alpha:
                        edgecounterX += 1
                    cond_setX = np.concatenate((removedX, Y.values[0]), axis=1)
                    valResX, pvalResX = RFDC(np.array(X.values[0][:, var1:var1 + 1]), np.array(X.values[0][:, var2:var2 + 1]),
                                                             Z=np.array(cond_setX),random_state=random_state)
                    if pvalResX < alpha:
                        edgecounterResX += 1
            dict['number of nodes X'] = len(X.var_names)
            dict[' max number of edges X'] = max_edgenumberX
            dict['number of edges X'] = edgecounterX
            dict['number of edges Res X'] = edgecounterResX
            dict['edge density of X'] = edgecounterX / max_edgenumberX
            dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
            dict['tested regions'] = type
            return dict
        if type == 'minimal':
            ## implementation of the minimal version of the test that decides based on d(X|Y) alone where the conditioning group
            ## is chosed to be the smaller one (or randomly if groups are of equal size)
            if len(Y.var_names) == len(X.var_names):
                random_choice = np.random.binomial(1, 0.5)
            if len(Y.var_names) > len(X.var_names) or random_choice == 1:
                edgecounterY = 0
                edgecounterResY = 0
                max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
                for var1 in range(len(Y.var_names)):
                    for var2 in range(var1 + 1, len(Y.var_names)):
                        removedY = np.delete(Y.values[0], (var1, var2), 1)
                        valY, pvalY = RFDC(np.array(Y.values[0][:, var1:var1 + 1]), np.array(Y.values[0][:, var2:var2 + 1]),
                                                           Z=removedY,random_state=random_state)
                        if pvalY < alpha:
                            edgecounterY += 1
                        cond_setY = np.concatenate((removedY, X.values[0]), axis=1)
                        valResY, pvalResY = RFDC(np.array(Y.values[0][:, var1:var1 + 1]), np.array(Y.values[0][:, var2:var2 + 1]),
                                                                 Z=cond_setY,random_state=random_state)
                        if pvalResY < alpha:
                            edgecounterResY += 1
                dict['number of nodes Y'] = len(Y.var_names)
                dict[' max number of edges Y'] = max_edgenumberY
                dict['number of edges Y'] = edgecounterY
                dict['number of edges Res Y'] = edgecounterResY
                dict['edge density of Y'] = edgecounterY / max_edgenumberY
                dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
                dict['tested region'] = 'Y'
                return dict
            else:
                edgecounterX = 0
                max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
                edgecounterResX = 0
                for var1 in range(len(X.var_names)):
                    for var2 in range(var1 + 1, len(X.var_names)):
                        removedX = np.delete(X.values[0], (var1, var2), 1)
                        valX, pvalX = RFDC(np.array(X.values[0][:, var1:var1 + 1]), np.array(X.values[0][:, var2:var2 + 1]),
                                                           Z=removedX,random_state=random_state)
                        if pvalX < alpha:
                            edgecounterX += 1
                        cond_setX = np.concatenate((removedX, Y.values[0]), axis=1)
                        valResX, pvalResX = RFDC(np.array(X.values[0][:, var1:var1 + 1]), np.array(X.values[0][:, var2:var2 + 1]),
                                                                 Z=cond_setX,random_state=random_state)
                        if pvalResX < alpha:
                            edgecounterResX += 1
                dict['number of nodes X'] = len(X.var_names)
                dict[' max number of edges X'] = max_edgenumberX
                dict['number of edges X'] = edgecounterX
                dict['number of edges Res X'] = edgecounterResX
                dict['edge density of X'] = edgecounterX / max_edgenumberX
                dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
                dict['tested region'] = 'X'
                return dict
    else:
        if CI_test_method == 'ParCorr':
            CI_test =ParCorr()
        if CI_test_method == 'GPDC':
            CI_test =GPDCtorch()
        if CI_test_method == 'CMIknn':
            CI_test =CMIknn()
        dict = {}
        if linear == True:
            Regression = regression(X, Y, fit_intercept=fit_intercept)
            if type == 'both':
                residualsY = Regression['X_to_Y'].residuals.values[0]
                edgecounterY = 0
                edgecounterResY = 0
                max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
                max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
                for var1 in range(len(Y.var_names)):
                    for var2 in range(var1 + 1, len(Y.var_names)):
                        removedY = np.delete(Y.values[0], (var1, var2), 1)
                        valY, pvalY = CI_test.run_test_raw(Y.values[0][:, var1:var1 + 1], Y.values[0][:, var2:var2 + 1], z=removedY)
                        if pvalY < alpha:
                            edgecounterY += 1
                        removedResY = np.delete(residualsY, (var1, var2), 1)
                        valResY, pvalResY = CI_test.run_test_raw(residualsY[:, var1:var1 + 1], residualsY[:, var2:var2 + 1],
                                                                 z=removedResY)
                        if pvalResY < alpha:
                            edgecounterResY += 1
                dict['number of nodes Y'] = len(Y.var_names)
                dict[' max number of edges Y'] = max_edgenumberY
                dict['number of edges Y'] = edgecounterY
                dict['number of edges Res Y'] = edgecounterResY
                dict['edge density of Y'] = edgecounterY / max_edgenumberY
                dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
                residualsX = Regression['Y_to_X'].residuals.values[0]
                edgecounterX = 0
                edgecounterResX = 0
                for var1 in range(len(X.var_names)):
                    for var2 in range(var1 + 1, len(X.var_names)):
                        removedX = np.delete(X.values[0], (var1, var2), 1)
                        valX, pvalX = CI_test.run_test_raw(X.values[0][:, var1:var1 + 1], X.values[0][:, var2:var2 + 1], z=removedX)
                        if pvalX < alpha:
                            edgecounterX += 1
                        removedResX = np.delete(residualsX, (var1, var2), 1)
                        valResX, pvalResX = CI_test.run_test_raw(residualsX[:, var1:var1 + 1], residualsX[:, var2:var2 + 1],
                                                                 z=removedResX)
                        if pvalResX < alpha:
                            edgecounterResX += 1
                dict['number of nodes X'] = len(X.var_names)
                dict[' max number of edges X'] = max_edgenumberX
                dict['number of edges X'] = edgecounterX
                dict['number of edges Res X'] = edgecounterResX
                dict['edge density of X'] = edgecounterX / max_edgenumberX
                dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
                dict['tested regions'] = type
                return dict
            if type == 'minimal':
                if len(Y.var_names) == len(X.var_names):
                    random_choice = np.random.binomial(1,0.5)
                if len(Y.var_names) > len(X.var_names) or random_choice == 1:
                    residualsY = Regression['X_to_Y'].residuals.values[0]
                    edgecounterY = 0
                    edgecounterResY = 0
                    max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
                    for var1 in range(len(Y.var_names)):
                        for var2 in range(var1 + 1, len(Y.var_names)):
                            removedY = np.delete(Y.values[0], (var1, var2), 1)
                            valY, pvalY = CI_test.run_test_raw(Y.values[0][:, var1:var1 + 1], Y.values[0][:, var2:var2 + 1],
                                                               z=removedY)
                            if pvalY < alpha:
                                edgecounterY += 1
                            removedResY = np.delete(residualsY, (var1, var2), 1)
                            valResY, pvalResY = CI_test.run_test_raw(residualsY[:, var1:var1 + 1], residualsY[:, var2:var2 + 1],
                                                                     z=removedResY)
                            if pvalResY < alpha:
                                edgecounterResY += 1
                    dict['number of nodes Y'] = len(Y.var_names)
                    dict[' max number of edges Y'] = max_edgenumberY
                    dict['number of edges Y'] = edgecounterY
                    dict['number of edges Res Y'] = edgecounterResY
                    dict['edge density of Y'] = edgecounterY / max_edgenumberY
                    dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
                    dict['tested region'] = 'Y'
                    return dict
                else:
                    residualsX = Regression['Y_to_X'].residuals.values[0]
                    edgecounterX = 0
                    max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
                    edgecounterResX = 0
                    for var1 in range(len(X.var_names)):
                        for var2 in range(var1 + 1, len(X.var_names)):
                            removedX = np.delete(X.values[0], (var1, var2), 1)
                            valX, pvalX = CI_test.run_test_raw(X.values[0][:, var1:var1 + 1], X.values[0][:, var2:var2 + 1], z=removedX)
                            if pvalX < alpha:
                                edgecounterX += 1
                            removedResX = np.delete(residualsX, (var1, var2), 1)
                            valResX, pvalResX = CI_test.run_test_raw(residualsX[:, var1:var1 + 1], residualsX[:, var2:var2 + 1],
                                                                     z=removedResX)
                            if pvalResX < alpha:
                                edgecounterResX += 1
                    dict['number of nodes X'] = len(X.var_names)
                    dict[' max number of edges X'] = max_edgenumberX
                    dict['number of edges X'] = edgecounterX
                    dict['number of edges Res X'] = edgecounterResX
                    dict['edge density of X'] = edgecounterX / max_edgenumberX
                    dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
                    dict['tested region'] = 'X'
                    return dict
        else:
            if type == 'both':
                edgecounterY = 0
                edgecounterResY = 0
                max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
                max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
                for var1 in range(len(Y.var_names)):
                    for var2 in range(var1 + 1, len(Y.var_names)):
                        removedY = np.delete(Y.values[0], (var1, var2), 1)
                        valY, pvalY = CI_test.run_test_raw(Y.values[0][:, var1:var1 + 1], Y.values[0][:, var2:var2 + 1], z=removedY)
                        if pvalY < alpha:
                            edgecounterY += 1
                        cond_setY = np.concatenate((removedY,X.values[0]),axis=1)
                        valResY, pvalResY = CI_test.run_test_raw(Y.values[0][:, var1:var1 + 1], Y.values[0][:, var2:var2 + 1],
                                                                 z=cond_setY)
                        if pvalResY < alpha:
                            edgecounterResY += 1
                dict['number of nodes Y'] = len(Y.var_names)
                dict[' max number of edges Y'] = max_edgenumberY
                dict['number of edges Y'] = edgecounterY
                dict['number of edges Res Y'] = edgecounterResY
                dict['edge density of Y'] = edgecounterY / max_edgenumberY
                dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
                edgecounterX = 0
                edgecounterResX = 0
                for var1 in range(len(X.var_names)):
                    for var2 in range(var1 + 1, len(X.var_names)):
                        removedX = np.delete(X.values[0], (var1, var2), 1)
                        valX, pvalX = CI_test.run_test_raw(X.values[0][:, var1:var1 + 1], X.values[0][:, var2:var2 + 1], z=removedX)
                        if pvalX < alpha:
                            edgecounterX += 1
                        cond_setX = np.concatenate((removedX, Y.values[0]), axis=1)
                        valResX, pvalResX = CI_test.run_test_raw(X.values[0][:, var1:var1 + 1], X.values[0][:, var2:var2 + 1],
                                                                 z=cond_setX)
                        if pvalResX < alpha:
                            edgecounterResX += 1
                dict['number of nodes X'] = len(X.var_names)
                dict[' max number of edges X'] = max_edgenumberX
                dict['number of edges X'] = edgecounterX
                dict['number of edges Res X'] = edgecounterResX
                dict['edge density of X'] = edgecounterX / max_edgenumberX
                dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
                dict['tested regions'] = type
                return dict
            if type == 'minimal':
                ## implementation of the minimal version of the test that decides based on d(X|Y) alone where the conditioning group
                ## is chosed to be the smaller one (or randomly if groups are of equal size)
                if len(Y.var_names) == len(X.var_names):
                    random_choice = np.random.binomial(1,0.5)
                if len(Y.var_names) > len(X.var_names) or random_choice == 1:
                    edgecounterY = 0
                    edgecounterResY = 0
                    max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
                    for var1 in range(len(Y.var_names)):
                        for var2 in range(var1 + 1, len(Y.var_names)):
                            removedY = np.delete(Y.values[0], (var1, var2), 1)
                            valY, pvalY = CI_test.run_test_raw(Y.values[0][:, var1:var1 + 1], Y.values[0][:, var2:var2 + 1],
                                                               z=removedY)
                            if pvalY < alpha:
                                edgecounterY += 1
                            cond_setY = np.concatenate((removedY, X.values[0]), axis=1)
                            valResY, pvalResY = CI_test.run_test_raw(Y.values[0][:, var1:var1 + 1], Y.values[0][:, var2:var2 + 1],
                                                                     z=cond_setY)
                            if pvalResY < alpha:
                                edgecounterResY += 1
                    dict['number of nodes Y'] = len(Y.var_names)
                    dict[' max number of edges Y'] = max_edgenumberY
                    dict['number of edges Y'] = edgecounterY
                    dict['number of edges Res Y'] = edgecounterResY
                    dict['edge density of Y'] = edgecounterY / max_edgenumberY
                    dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
                    dict['tested region'] = 'Y'
                    return dict
                else:
                    edgecounterX = 0
                    max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
                    edgecounterResX = 0
                    for var1 in range(len(X.var_names)):
                        for var2 in range(var1 + 1, len(X.var_names)):
                            removedX = np.delete(X.values[0], (var1, var2), 1)
                            valX, pvalX = CI_test.run_test_raw(X.values[0][:, var1:var1 + 1], X.values[0][:, var2:var2 + 1],
                                                               z=removedX)
                            if pvalX < alpha:
                                edgecounterX += 1
                            cond_setX = np.concatenate((removedX, Y.values[0]), axis=1)
                            valResX, pvalResX = CI_test.run_test_raw(X.values[0][:, var1:var1 + 1], X.values[0][:, var2:var2 + 1],
                                                                     z=cond_setX)
                            if pvalResX < alpha:
                                edgecounterResX += 1
                    dict['number of nodes X'] = len(X.var_names)
                    dict[' max number of edges X'] = max_edgenumberX
                    dict['number of edges X'] = edgecounterX
                    dict['number of edges Res X'] = edgecounterResX
                    dict['edge density of X'] = edgecounterX / max_edgenumberX
                    dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
                    dict['tested region'] = 'X'
                    return dict


def conditioning_ind_test_with_PC(X, Y, alpha=0.01, type='both', CI_test_method='ParCorr', max_sep_set = None, fit_intercept = False, linear_interactions = True):
    '''
    Implementation of 2G-VecCI.PC
    runs sparsity based independent test of regions X and Y using the PC algorithm. To include one group in the conditioning set
    in every CI test, we first perform a regression step (either OLS or Gaussian process regression). PC is then applied to the
    residuals
    
    Parameters:
        X : pp.Dataframes containing regional variables,
        alpha : floating number, significance level for conditional independence testing
        type : string, either 'both' or 'minimal'. Conditional Independences are tested either on both
            regions or on the smaller region
        CI_test_method : The conditional independence test. Options: 'ParCorr', 'GPDC', see the documentation
            of Tigramite for details on the implementations of these tests
        max_sep_set : provides upper bound for separating sets.
        fit_intercept : specifies whether an intercept should be fitted in the regression
        linear_interactions : if True, the conditioning regression is OLS, if False it is a Gaussian process regression
        
    Returns:
        dictionary : describing number of detected edges on regions and their residuals as well as sparsity,
        measured as number of detected edges/number of edges on fully connected graph
    '''
    if linear_interactions == True:
        Regression = regression(X, Y, fit_intercept=fit_intercept)
    else:
        Regression = GP_regression(X, Y)
    
    if CI_test_method == 'ParCorr':
        CI_test =ParCorr()
    elif CI_test_method == 'GPDC':
        CI_test =GPDCtorch()
    elif CI_test_method == 'CMIknn':
       CI_test =CMIknn()
    dict = {}

    if type == 'both':
        residualsY = Regression['X_to_Y'].residuals
        max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
        max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
        ResY_pcmci_parcorr = PCMCI(dataframe=residualsY, cond_ind_test=CI_test)
        resultsResY = ResY_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
        edgecounterResY = edge_count(ResY_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsResY['p_matrix'],alpha_level=alpha, tau_min=0, tau_max=0))
        Y_pcmci_parcorr = PCMCI(dataframe=Y, cond_ind_test=CI_test)
        resultsY = Y_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
        edgecounterY = edge_count(Y_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsY['p_matrix'], alpha_level=alpha, tau_min=0, tau_max=0))
        dict['number of nodes Y'] = len(Y.var_names)
        dict[' max number of edges Y'] = max_edgenumberY
        dict['number of edges Y'] = edgecounterY
        dict['number of edges Res Y'] = edgecounterResY
        dict['edge density of Y'] = edgecounterY / max_edgenumberY
        dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY

        residualsX = Regression['Y_to_X'].residuals
        ResX_pcmci_parcorr = PCMCI(dataframe=residualsX, cond_ind_test=CI_test)
        resultsResX = ResX_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
        edgecounterResX = edge_count(ResX_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsResX['p_matrix'], alpha_level=alpha, tau_min=0,tau_max=0))
        X_pcmci_parcorr = PCMCI(dataframe=X, cond_ind_test=CI_test)
        resultsX = X_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
        edgecounterX = edge_count(
            X_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsX['p_matrix'], alpha_level=alpha, tau_min=0, tau_max=0))
        dict['number of nodes X'] = len(X.var_names)
        dict[' max number of edges X'] = max_edgenumberX
        dict['number of edges X'] = edgecounterX
        dict['number of edges Res X'] = edgecounterResX
        dict['edge density of X'] = edgecounterX / max_edgenumberX
        dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
        dict['tested regions'] = type
        return dict
    
    if type == 'minimal':
        if len(Y.var_names) == len(X.var_names):
            random_choice = np.random.binomial(1,0.5)
        if len(Y.var_names) > len(X.var_names) or random_choice == 1:
            residualsY = Regression['X_to_Y'].residuals
            max_edgenumberY = len(Y.var_names) * (len(Y.var_names) - 1) / 2
            ResY_pcmci_parcorr = PCMCI(dataframe=residualsY, cond_ind_test=CI_test)
            resultsResY = ResY_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
            edgecounterResY = edge_count(
                ResY_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsResY['p_matrix'], alpha_level=alpha,
                                                          tau_min=0, tau_max=0))
            Y_pcmci_parcorr = PCMCI(dataframe=Y, cond_ind_test=CI_test)
            resultsY = Y_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
            edgecounterY = edge_count(
                Y_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsY['p_matrix'], alpha_level=alpha, tau_min=0,
                                                       tau_max=0))
            dict['number of nodes Y'] = len(Y.var_names)
            dict[' max number of edges Y'] = max_edgenumberY
            dict['number of edges Y'] = edgecounterY
            dict['number of edges Res Y'] = edgecounterResY
            dict['edge density of Y'] = edgecounterY / max_edgenumberY
            dict['edge density of Res Y'] = edgecounterResY / max_edgenumberY
            return dict
        else:
            residualsX = Regression['Y_to_X'].residuals
            max_edgenumberX = len(X.var_names) * (len(X.var_names) - 1) / 2
            ResX_pcmci_parcorr = PCMCI(dataframe=residualsX, cond_ind_test=CI_test)
            resultsResX = ResX_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
            edgecounterResX = edge_count(
                ResX_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsResX['p_matrix'], alpha_level=alpha,
                                                          tau_min=0, tau_max=0))
            X_pcmci_parcorr = PCMCI(dataframe=X, cond_ind_test=CI_test)
            resultsX = X_pcmci_parcorr.run_pcalg_non_timeseries_data(pc_alpha=alpha, max_conds_dim=max_sep_set)
            edgecounterX = edge_count(
                X_pcmci_parcorr.get_graph_from_pmatrix(p_matrix=resultsX['p_matrix'], alpha_level=alpha, tau_min=0,
                                                       tau_max=0))
            dict['number of nodes X'] = len(X.var_names)
            dict[' max number of edges X'] = max_edgenumberX
            dict['number of edges X'] = edgecounterX
            dict['number of edges Res X'] = edgecounterResX
            dict['edge density of X'] = edgecounterX / max_edgenumberX
            dict['edge density of Res X'] = edgecounterResX / max_edgenumberX
            return dict

def edge_count(graph):
    # function for counting edges in graph
    count = 0
    for i in graph:
        for edge in i:
            if edge != '':
                count +=1
    return int(count/2)


def trace_method(X,Y):
    '''
    Implementation of the trace method of (Janzing et al, 2010, Telling cause from effect based on high- dimensional observations).
    
    Parameters:
        X : tigramite dataframe of form (T,N) where T is sample and N is group size
        Y : tigramite dataframe of form (T,N') where T is sample and N' is group size
    
    Returns:
        delta values as defined in the paper.
    '''
    Xt = X.values[0].T
    Yt = Y.values[0].T
    CovX = np.cov(Xt)
    CovY = np.cov(Yt)
    XandY = np.concatenate((Xt, Yt), axis=0)
    sizeX = CovX.shape[0]
    sizeY = CovY.shape[0]
    CovXY = np.cov(XandY)[:sizeX, sizeX:]
    CovYX = CovXY.T
    pinvX = np.linalg.pinv(CovX)
    pinvY = np.linalg.pinv(CovY)
    A1 = np.matmul(CovYX, pinvX)
    A2 = np.matmul(CovXY, pinvY)
    rank1 = np.trace(np.matmul(CovX, pinvX))
    rank2 = np.trace(np.matmul(CovY, pinvY))
    delta1 = np.log(np.trace(np.matmul(np.matmul(A1, CovX), A1.T)) / sizeX) - np.log(
        np.trace(np.matmul(A1, A1.T)) / rank1) - np.log(np.trace(CovX) / sizeX)
    delta2 = np.log(np.trace(np.matmul(np.matmul(A2, CovY), A2.T)) / sizeY) - np.log(
        np.trace(np.matmul(A2, A2.T)) / rank2) - np.log(np.trace(CovY) / sizeY)
    return delta1, delta2


def identify_causal_direction(X: pp.DataFrame , Y: pp.DataFrame, alpha=0.01,
                              type = 'both', CI_test_method='ParCorr', ambiguity = None,
                              test = 'full', max_sep_set = None, linear = True,
                              fit_intercept = False, random_state = None):
    '''
    Function that executes either 2G-VecCI.PC or 2G-VecCi.Full and outputs the causal direction as well as the inference criterion
    and details on edge counts and densities
    
    Parameters:
        X : tigramite dataframe of form (T,N) where T is sample and N is group size
        Y : tigramite dataframe of form (T,N') where T is sample and N' is group size
        alpha : significance level of CI tests
        type : string, either 'minimal' or 'both' for one-sided or two-sided test
        CI_test_method : conditional independence test method, default ParCorr
        ambiguity : ambiguity level as specified in submitted paper, if None chosen to be alpha
        test : string, either 'full' for 2G-VecCI.Full or 'PC' for 2G-VecCI.PC
        max_sep_set : maximal size of d-separation set in case 2G-VecCI.PC is used
        linear : boolean, tests for linear or non-linear interactions
        fit_intercept : boolean, indicates whether an intercept should be fitted in regression methods, default is False
    
    Returns:
        comparison : string indicating the inferred causal direction
        cause_criterion : Crit, as specified in the submitted paper
        test_results : details on edge densities and counts
    '''

    if ambiguity == None:
        ambiguity = alpha

    if test == 'full':
        test_results = full_conditioning_ind_test(X,Y,alpha,type,CI_test_method=CI_test_method, linear = linear, fit_intercept=fit_intercept, random_state=random_state)
    elif test == 'PC':
        test_results = conditioning_ind_test_with_PC(X, Y, alpha, type, CI_test_method=CI_test_method, max_sep_set=max_sep_set, linear_interactions=linear)
    
    if type == 'both':
        X_criterion = test_results['edge density of X'] - test_results['edge density of Res X']
        Y_criterion = test_results['edge density of Y'] - test_results['edge density of Res Y']
        if abs(X_criterion - Y_criterion) < ambiguity:
            comparison = 'Causal direction can not be inferred'
        elif Y_criterion - X_criterion > ambiguity:
            comparison = 'X is the cause of Y'
        elif X_criterion - Y_criterion  > ambiguity:
            comparison = 'Y is the cause of X'
        return comparison, X_criterion, Y_criterion, test_results
    elif type == 'minimal':
        cause_criterion = test_results['edge density of %s'%(test_results['tested region'])] - test_results['edge density of Res %s'%(test_results['tested region'])]
        if abs(cause_criterion) < ambiguity:
            comparison = 'Causal direction can not be inferred'
        elif cause_criterion > ambiguity:
            comparison = '%s is the effect variable'%(test_results['tested region'])
        elif cause_criterion < -ambiguity:
            comparison = '%s is the cause variable'%(test_results['tested region'])
        return comparison, cause_criterion, test_results

def identify_causal_direction_trace_method(X,Y,eps=0.1):
    ##function that executes trace method.
    # OUTPUT: same format as identify_causal_direction
    test_results = trace_method(X,Y)
    X_criterion = test_results[0]
    Y_criterion = test_results[1]
    if  abs(Y_criterion) > (1.0+eps)*abs(X_criterion):
        comparison = 'X is the cause of Y'
    elif abs(X_criterion) > (1.0+eps)*abs(Y_criterion):
        comparison = 'Y is the cause of X'
    else:
        comparison = 'Causal direction can not be inferred'
    return comparison, X_criterion, Y_criterion


def total_graph_causal_direction(graph, len_var_1, len_var_2):
    # Helper functon that counts links from X TO Y that
    # are oriented rightly (right_causal) AND wrongly (wrong_causal)
    # Returns Arrow counts right_causal and wrong_causal

    right_causal = 0
    wrong_causal = 0
    num_links = 0

    for i in range(len_var_1):
        for j in range(len_var_1, len_var_1 + len_var_2):
            if (graph[i][j] != ['']):
                num_links = num_links + 1
                if (graph[i][j] == ['-->']):
                    right_causal = right_causal + 1
                elif (graph[i][j] == ['<--']):
                    wrong_causal = wrong_causal + 1

    total_edges = len_var_1 * len_var_2

    return [right_causal, wrong_causal,total_edges]


def vanilla_pc(X, Y, alpha=0.01, CI_test_method='ParCorr'):
    # Runs vanilla PC over groups X and Y concatenated into single dataframe
    # and returns the righlty and wrongly oriented arrow counts (arrow_count[0] and arrow_count[1] resp.)
    # and total number of links between groups

    X_vector = X.values[0]
    Y_vector = Y.values[0]
    data_full = np.concatenate((X_vector, Y_vector), axis=1)
    dataframe_full = pp.DataFrame(data_full)

    if CI_test_method == 'ParCorr':
        CI_test = ParCorr()
    if CI_test_method == 'GPDC':
        CI_test = GPDCtorch()

    pcmci_full = PCMCI(
        dataframe=dataframe_full,
        cond_ind_test=CI_test,
        verbosity=0)

    results = pcmci_full.run_pcalg_non_timeseries_data(pc_alpha=alpha)

    graph = np.array(results['graph'])

    len_var_1 = len(X.var_names)
    len_var_2 = len(Y.var_names)

    arrow_counter = total_graph_causal_direction(graph, len_var_1, len_var_2)

    return arrow_counter


def identify_causal_direction_vanilla_pc(X,Y, alpha=0.01, CI_test_method='ParCorr', ambiguity=None):
    # Given ambiguity parameter, returns whether and which causal direction is inferred

    arrow_counter = vanilla_pc(X,Y, alpha, CI_test_method)

    if ambiguity == None:
        ambiguity = alpha

    if arrow_counter[2]== 0:
        comparison = 'Causal direction can not be inferred'
    elif abs((arrow_counter[0] - arrow_counter[1]) / arrow_counter[2]) < ambiguity:
        comparison = 'Causal direction can not be inferred'
    elif arrow_counter[0] > arrow_counter[1]:
        comparison = 'X is the cause of Y'
    else:
        comparison = 'Y is the cause of X'

    return comparison


def average_lingam(X, Y):
    # Function that averages over groups X and Y resp. and
    # calculates causal order between the averages using Direct Lingam (Shimizu et al. '11)

    X_vector = X.values[0]
    Y_vector = Y.values[0]
    T, N = X_vector.shape

    data_average = np.array([[X_vector.mean(axis=1)[i], Y_vector.mean(axis=1)[i]] for i in range(T)])
    dataframe_average = pp.DataFrame(data_average)

    model = lingam.DirectLiNGAM()
    model.fit(dataframe_average.values)

    # label '0' corresponds to cause and '1' to effect so the correct causal order is [0,1]
    wrong_causal, right_causal = model.causal_order_

    if wrong_causal == 1:
        comparison = 'Y is the cause of X'
    else:
        comparison = 'X is the cause of Y'

    return comparison

def compute_edge_number(N, edge_sparsity):
    return int(edge_sparsity*N*(N-1)/2)


def summary(configurations, random_state = None, test = 'full',ambiguity = None, CI_test_method = 'ParCorr', max_sep_set = None, linear = True, noise_type = 'gaussian', standardize = True):
    ## function for running 2G-VecCI on randomly generated data with linear interactions
    ##INPUT:
    ## configurations = list of parameter specifications generated by the set_parameter file
    ## test = string, either 'full' for 2G-VecCI.Full or 'PC' for 2G-VecCI.PC
    ##CI_test_method = conditional independence test method, default ParCorr
    ## max_sep_set = maximal size of d-separation set in case 2G-VecCI.PC is used
    ##linear = boolean, tests for linear or non-linear interactions, DOES NOT REFER to data generation which is always linear in this function
    ## noise_type = string, either 'gaussian' or 'mixed' for either Gaussian or Gaussian and Weibull noises
    ##OUTPUT:
    ##summary_dict = dictionary providing a count of correct/wrong/undecided inferences per parameter configuration
    ##time_list = list of computation times

    if random_state == None:
        random_state = np.random
    summary_dict = {}
    time_list = []
    for config in configurations:
        print(f'Running configuration {config}')
        summary_dict[config] = [0, 0, 0]
        for i in range(config[-1]):
            edge_number_cause = compute_edge_number(config[1], config[3])
            edge_number_effect = compute_edge_number(config[2], config[4])
            model = biv_model(cause_no=config[1], effect_no=config[2], cause_link_no=edge_number_cause,
                              effect_link_no=edge_number_effect, sparsity=config[5], samplesize=config[0],
                              random_state=random_state, noise_type=noise_type)
            model.build_model()
            start = time.time()
            if standardize == True:
                causal_inference_in_model = identify_causal_direction(X=standardize_DataFrame(model.cause), Y=standardize_DataFrame(model.effect), alpha=config[8],
                                                                      type=config[7], CI_test_method=CI_test_method,
                                                                      ambiguity=ambiguity, test=test,
                                                                      max_sep_set=max_sep_set, linear=linear, random_state=random_state)
            else:
                causal_inference_in_model = identify_causal_direction(X=model.cause, Y=model.effect, alpha=config[8],
                                                                  type=config[7], CI_test_method=CI_test_method,ambiguity=ambiguity, test = test, max_sep_set = max_sep_set, linear = linear, random_state=random_state)
            time_list.append(time.time()-start)
            if causal_inference_in_model[0] == 'X is the cause of Y' or causal_inference_in_model[
                0] == 'X is the cause variable' or causal_inference_in_model[0] == 'Y is the effect variable':
                summary_dict[config][0] += 1
            if causal_inference_in_model[0] == 'Y is the cause of X' or causal_inference_in_model[
                0] == 'Y is the cause variable' or causal_inference_in_model[0] == 'X is the effect variable':
                summary_dict[config][1] += 1
            if causal_inference_in_model[0] == 'Causal direction can not be inferred':
                summary_dict[config][2] += 1
    return summary_dict, time_list

def summary_trace_method(configurations, random_state = None, eps = 0.1, standardize = True):
    ##  as in summary but with 2GVecCI replaced by the trace method, eps = ambiguity for trace method
    if random_state == None:
        random_state = np.random
    summary_dict = {}
    for config in configurations:
        summary_dict[config] = [0, 0, 0]
        for i in range(config[-1]):
            edge_number_cause = compute_edge_number(config[1], config[3])
            edge_number_effect = compute_edge_number(config[2], config[4])
            model = biv_model(cause_no=config[1], effect_no=config[2], cause_link_no=edge_number_cause,
                              effect_link_no=edge_number_effect, sparsity=config[5], samplesize=config[0],
                              random_state=random_state)
            model.build_model()
            if standardize == True:
                causal_inference_in_model = identify_causal_direction_trace_method(X=standardize_DataFrame(model.cause), Y=standardize_DataFrame(model.effect), eps = eps)
            else:
                causal_inference_in_model = identify_causal_direction_trace_method(X=model.cause, Y=model.effect, eps = eps)
            if causal_inference_in_model[0] == 'X is the cause of Y' or causal_inference_in_model[
                0] == 'X is the cause variable' or causal_inference_in_model[0] == 'Y is the effect variable':
                summary_dict[config][0] += 1
            if causal_inference_in_model[0] == 'Y is the cause of X' or causal_inference_in_model[
                0] == 'Y is the cause variable' or causal_inference_in_model[0] == 'X is the effect variable':
                summary_dict[config][1] += 1
            if causal_inference_in_model[0] == 'Causal direction can not be inferred':
                summary_dict[config][2] += 1
    return summary_dict

def summary_vanilla_PC_linear(configurations, random_state = None, ambiguity = None, standardize = False):
    ##  as in summary but with 2GVecCI replaced by vanilla PC, eps = ambiguity for trace method
    if random_state == None:
        random_state = np.random
    summary_dict = {}
    for config in configurations:
        summary_dict[config] = [0, 0, 0]
        for i in range(config[-1]):
            edge_number_cause = compute_edge_number(config[1], config[3])
            edge_number_effect = compute_edge_number(config[2], config[4])
            model = biv_model(cause_no=config[1], effect_no=config[2], cause_link_no=edge_number_cause,
                              effect_link_no=edge_number_effect, sparsity=config[5], samplesize=config[0],
                              random_state=random_state)
            model.build_model()
            if standardize == True:
                causal_inference_in_model = identify_causal_direction_vanilla_pc(X=standardize_DataFrame(model.cause), Y=standardize_DataFrame(model.effect), ambiguity=ambiguity)
            else:
                causal_inference_in_model = identify_causal_direction_vanilla_pc(X=model.cause, Y=model.effect, ambiguity=ambiguity)
            if causal_inference_in_model == 'X is the cause of Y' or causal_inference_in_model[
                0] == 'X is the cause variable' or causal_inference_in_model[0] == 'Y is the effect variable':
                summary_dict[config][0] += 1
            if causal_inference_in_model == 'Y is the cause of X' or causal_inference_in_model[
                0] == 'Y is the cause variable' or causal_inference_in_model[0] == 'X is the effect variable':
                summary_dict[config][1] += 1
            if causal_inference_in_model == 'Causal direction can not be inferred':
                summary_dict[config][2] += 1
    return summary_dict


def summary_vanilla_PC_quadratic(configurations, random_state = None, ambiguity = None, standardize = False):
    ##  as in summary but with 2GVecCI replaced by vanilla PC, eps = ambiguity for trace method
    if random_state == None:
        random_state = np.random
    summary_dict = {}
    for config in configurations:
        summary_dict[config] = [0, 0, 0]
        for i in range(config[-1]):
            edge_number_cause = compute_edge_number(config[1], config[3])
            edge_number_effect = compute_edge_number(config[2], config[4])
            model = biv_model(cause_no=config[1], effect_no=config[2], cause_link_no=edge_number_cause,
                              effect_link_no=edge_number_effect, sparsity=config[5], samplesize=config[0],
                              random_state=random_state)
            model.build_quadratic_interaction_model()
            if standardize == True:
                causal_inference_in_model = identify_causal_direction_vanilla_pc(X=standardize_DataFrame(model.cause), Y=standardize_DataFrame(model.effect),CI_test_method='GPDC', ambiguity=ambiguity)
            else:
                causal_inference_in_model = identify_causal_direction_vanilla_pc(X=model.cause, Y=model.effect,CI_test_method='GPDC', ambiguity=ambiguity)
            if causal_inference_in_model == 'X is the cause of Y' or causal_inference_in_model[
                0] == 'X is the cause variable' or causal_inference_in_model[0] == 'Y is the effect variable':
                summary_dict[config][0] += 1
            if causal_inference_in_model == 'Y is the cause of X' or causal_inference_in_model[
                0] == 'Y is the cause variable' or causal_inference_in_model[0] == 'X is the effect variable':
                summary_dict[config][1] += 1
            if causal_inference_in_model == 'Causal direction can not be inferred':
                summary_dict[config][2] += 1
    return summary_dict

def summary_quadratic(configurations, random_state = None, test='full', ambiguity = None, CI_test_method = 'GPDC', max_sep_set = None, linear = 'no'):
    ##  as in summary but generated random quadratic interaction models
    if random_state == None:
        random_state = np.random
    summary_dict = {}
    for config in configurations:
        summary_dict[config] = [0, 0, 0]
        for i in range(config[-1]):
            edge_number_cause = compute_edge_number(config[1], config[3])
            edge_number_effect = compute_edge_number(config[2], config[4])
            model = biv_model(cause_no=config[1], effect_no=config[2], cause_link_no=edge_number_cause,
                              effect_link_no=edge_number_effect, sparsity=config[5], samplesize=config[0],
                              random_state=random_state)
            model.build_quadratic_interaction_model()
            causal_inference_in_model = identify_causal_direction(X=model.cause, Y=model.effect, alpha=config[8],
                                                                  type=config[7],ambiguity=ambiguity, CI_test_method=CI_test_method, test = test, max_sep_set = max_sep_set, linear = linear, random_state=random_state)
            if causal_inference_in_model[0] == 'X is the cause of Y' or causal_inference_in_model[
                0] == 'X is the cause variable' or causal_inference_in_model[0] == 'Y is the effect variable':
                summary_dict[config][0] += 1
            if causal_inference_in_model[0] == 'Y is the cause of X' or causal_inference_in_model[
                0] == 'Y is the cause variable' or causal_inference_in_model[0] == 'X is the effect variable':
                summary_dict[config][1] += 1
            if causal_inference_in_model[0] == 'Causal direction can not be inferred':
                summary_dict[config][2] += 1
    return summary_dict

####PLOTTING

def get_all_aux_parameters(plot_parameter, full_parameters):
    ##helper function to separate the parameter along which a plot is to happen and the remaining parameters
    aux_parameters = copy.deepcopy(full_parameters)
    del aux_parameters[plot_parameter]
    return aux_parameters

def get_fixed_aux_parameters(plot_parameter, full_parameters):
    ##helper function to list all auxillary parameters
    all_aux = get_all_aux_parameters(plot_parameter,full_parameters)
    aux_par_list = []
    for aux_config in itertools.product(*list(all_aux.values())):
        listed_aux_config = []
        for entry in aux_config:
            listed_aux_config.append([entry])
        fixed_dict = dict(zip(all_aux.keys(),listed_aux_config))
        aux_par_list.append(fixed_dict)
    return aux_par_list

def get_dict_for_plot(summary_dict, plot_parameter, full_parameters, aux_parameters = {}):
    ## produces a description string and a dictionary for which all parameters are fixed except
    ## for the plot parameter (string). Aux_parameters is a dictionary of lists.
    if aux_parameters == {}:
        description2 = 'Percentages across all choices for auxillary parameters'
    else:
        description2 = 'Aux:' + str(list(aux_parameters.values()))
    if aux_parameters == {}:
        aux_parameters = get_all_aux_parameters(plot_parameter,full_parameters)
    full_parameters_list = list(full_parameters.keys())
    description = 'Plot of ' + plot_parameter
    required_dict = copy.deepcopy(summary_dict)
    for config in summary_dict:
        for aux in aux_parameters.keys():
            if config[full_parameters_list.index(aux)] not in aux_parameters[aux]:
                if config in required_dict:
                    del required_dict[config]
    return description, description2, required_dict


def percentages_for_plots(plotting_info, plot_parameter, full_parameters, errors = False, resamplesize = 20,num_resamples = 500, random_state = None):
    ##plotting_info is a triple of two strings describing the plot and a dictionary
    ##with the summarized results for given parameter values
    ##returns a tuple of two strings and three dictionaries: one for the
    ##percentage of correct, wrong, and undecided inferences
    ## these percentages are averages across all choices of auxillary parameters
    ##if errors = True, post-hoc error bars are bootstrapped where resamplesize indicates the size of bootstrap samples and num_resamples
    ##indicates the number of bootstrap resamples
    if random_state==None:
        random_state = np.random
    plot_dict = plotting_info[2]
    correct_infs, wrong_infs, undecided, bootstrap_confidences_correct, bootstrap_confidences_wrong, bootstrap_confidences_undecided = {}, {}, {}, {}, {}, {}
    for val in full_parameters[plot_parameter]:
        correct_infs[val], wrong_infs[val], undecided[val] = 0, 0, 0
    for config in plot_dict:
        correct_infs[config[list(full_parameters.keys()).index(plot_parameter)]] += plot_dict[config][0]
        wrong_infs[config[list(full_parameters.keys()).index(plot_parameter)]] += plot_dict[config][1]
        undecided[config[list(full_parameters.keys()).index(plot_parameter)]] += plot_dict[config][2]
    for val in full_parameters[plot_parameter]:
        total = correct_infs[val] + wrong_infs[val] + undecided[val]
        if total != 0:
            correct_infs[val] /= total
            wrong_infs[val] /= total
            undecided[val] /= total
        if errors == True:
            bootstrap_expect_correct = []
            bootstrap_expect_wrong = []
            bootstrap_expect_undecided = []
            for i in range(num_resamples):
                resample = list(random_state.multinomial(resamplesize,[correct_infs[val],wrong_infs[val],undecided[val]],size=1)[0])
                bootstrap_expect_correct.append(resample[0]/resamplesize)
                bootstrap_expect_wrong.append(resample[1] / resamplesize)
                bootstrap_expect_undecided.append(resample[2] / resamplesize)
            #bootstrap_lower = [correct_infs[val] - 1.96*np.std(bootstrap_expect_correct),wrong_infs[val] - 1.96*np.std(bootstrap_expect_wrong),undecided[val] - 1.96*np.std(bootstrap_expect_undecided)]
            #bootstrap_upper = [correct_infs[val] + 1.96*np.std(bootstrap_expect_correct),wrong_infs[val] + 1.96*np.std(bootstrap_expect_wrong),undecided[val] + 1.96*np.std(bootstrap_expect_undecided)]
            #bootstrap_lower_error[val] = bootstrap_lower
            #bootstrap_upper_error[val] = bootstrap_upper
            bootstrap_confidences_correct[val] = 1.96*np.std(bootstrap_expect_correct)
            bootstrap_confidences_wrong[val] = 1.96 * np.std(bootstrap_expect_wrong)
            bootstrap_confidences_undecided[val] = 1.96 * np.std(bootstrap_expect_undecided)


    return plotting_info[0], plotting_info[1], correct_infs, wrong_infs, undecided, bootstrap_confidences_correct, bootstrap_confidences_wrong, bootstrap_confidences_undecided

def plot_perc_in_one(perc,plot_parameter, my_path = '', errors = False):
    ##function for plotting percentages in one plot
    if plot_parameter == 'interaction_density':
        plot_parameter = 'Interaction density'
    elif plot_parameter == 'cause_region_density':
        plot_parameter = 'Cause region density'
    elif plot_parameter == 'cause_region_size':
        plot_parameter = 'Group size'
    elif plot_parameter == 'samplesize':
        plot_parameter = 'Number of samples'
    x= list(perc[2].keys())
    y = list(perc[2].values())
    y2 = list(perc[3].values())
    y3 = list(perc[4].values())

    mpl.style.use('Solarize_Light2')
    if errors == True:
        errors_y = list(perc[5].values())
        errors_y2 = list(perc[6].values())
        errors_y3 = list(perc[7].values())
        plt.errorbar(x,y,yerr=errors_y, capsize=5., marker = 'o')
        plt.errorbar(x, y2, yerr=errors_y2, capsize=5., marker = 'o')
        plt.errorbar(x, y3, yerr=errors_y3,capsize=5., marker = 'o')
    else:
        plt.plot(x, y, x, y2, x, y3, marker='o')
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.1, 0.1))
    #plt.title(perc[0])
    plt.xlabel(plot_parameter, fontsize = 14)
    plt.ylabel('Percentages', fontsize = 14)
    plt.legend(['correct inferences', 'wrong inferences', 'indeterminate'])
    filename = my_path + perc[0] + '_' + perc[1] + '.png'
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def direct_plotting(summary_dict, plot_parameter, full_parameters,aux_parameters ={}, my_path = '', errors = False):
     ##direct plotting function that executes the whole plotting routine in one function
    plotting_info = get_dict_for_plot(summary_dict, plot_parameter, full_parameters, aux_parameters)
    perc = percentages_for_plots(plotting_info, plot_parameter, full_parameters, errors=errors)
    plot_perc_in_one(perc, plot_parameter, my_path, errors=errors)

def split(a, n):
    k, m = len(a) // n, len(a) % n
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

# def refactor_string(string_list):
#     ##helpfunction for string refactoring when reading
#     help_config = []
#     for i in range(3):
#         help_config.append(int(string_list[i]))
#     for i in range(3,6):
#         help_config.append(float(string_list[i]))
#     help_config.append(tuple([float(string_list[6]),float(string_list[7])]))
#     help_config.append(string_list[8].replace("'",''))
#     help_config.append(float(string_list[9]))
#     help_config.append(int(string_list[10]))
#     return tuple(help_config)

def read_summary(mypath):
    ##function to read summaries from external file
    dict_list = []
    directory = os.fsencode(mypath)
    for filename in os.listdir(directory):
        #print(filename)
        if str(filename).startswith("b'samplesize") or str(filename).startswith("b'summary") :
            new_name = mypath + '/' + str(filename).replace("b'","").replace("'","")
            #print(new_name)
            with open(new_name, 'r') as dat:
                lines = dat.read().splitlines()
                lines_dict = {}
                for line in lines:
                    l = line.split(":")
                    if len(l) == 2:
                        lines_dict[l[0]] = ast.literal_eval(l[1])
                dict_list.append(lines_dict)
    summary_dict = {}
    for dict in dict_list:
        summary_dict.update(dict)
    new_summary_dict = {}
    for string in summary_dict.keys():
        string_list = list(string.replace(')', '').replace('(', '').replace(' ','').split(','))
        new_summary_dict[refactor_string(string_list)] = summary_dict[string]
    return new_summary_dict
