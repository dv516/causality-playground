from collections import defaultdict, deque

import numpy as np
from scipy.stats import t
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd


def t_critical_value(confidence_level: float, degrees_freedom: int) -> float:
    """
    Calculate the critical t-value for a given confidence level and degrees of freedom.

    Parameters:
    confidence_level (float): Desired confidence level (between 0 and 1).
    degrees_freedom (int): Degrees of freedom for the t-distribution.

    Returns:
    float: Critical t-value corresponding to the confidence level and degrees of freedom.
    """
    # Calculate the t-critical value using ppf (Percent Point Function)
    # The ppf function gives the value of t such that P(T <= t) = confidence_level
    t_crit = t.ppf(1 - (1 - confidence_level) / 2, df=degrees_freedom)
    return t_crit

def spearman_correlations_to_tstudent(correlations: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame of correlation coefficients to t-student values.

    Parameters:
    correlations (pd.DataFrame): DataFrame of correlation coefficients.

    Returns:
    pd.DataFrame: DataFrame of t-student values.
    """

    n = len(correlations)

    # Ensure the number of samples is greater than 2
    if n <= 2:
        raise ValueError("The number of samples must be greater than 2.")
    
    # Ensure all correlation values are between -1 and 1
    if not ((correlations >= -1).all().all() and (correlations <= 1).all().all()):
        raise ValueError("All correlation values must be between -1 and 1.")
    
    # Apply the formula to convert correlation coefficients to t-student values
    t_student_values = correlations.applymap(lambda r: r * np.sqrt(n - 2) / np.sqrt(1 - r**2))
    
    return t_student_values

def check_DAG(adj_mat):

    n = len(adj_mat)

    children = defaultdict(list)
    in_degree = defaultdict(int)

    for i in range(n):
        for j in range(n):
            if adj_mat[i,j]:
                children[i].append(j)
                in_degree[j] += 1

    queue = deque([node for node in range(n) if in_degree[node]==0])
    visited = []
    
    while queue:
        node = queue.popleft()
        visited.append(node)
        for neighbor in children[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return len(visited) == n

class Search():
    def __init__(self, dataframe):
        self.data = dataframe
        self.n = len(dataframe)
        self.weights = None
        self.sample_variance = dataframe.var()
        self.cols = len(dataframe.columns)

    # def _compute_empty_BIC(self, col):
    #     log_likelihood = -self.n/2*(np.log(2*np.pi)+np.log(self.mse)+1)
    #     return 

    def _evaluate_linear_BIC(self, adj_mat):

        if not check_DAG(adj_mat):
            raise ValueError('Adjacency matrix is not a valid DAG')

        total_BIC = 0.

        for i in range(len(self.cols)):
            parents = [self.cols[j] for j in range(len(self.cols))]
            if len(parents) == 0:
                pass



class ScikitLinearRegressor():
    def __init__(self, dataframe, cols=None, target_col=None):
        self.data = dataframe
       
        if target_col is None and cols is None:
            target_col = dataframe.columns[-1]
            cols = dataframe.columns[:-1]
        elif cols is None:
            self.target = dataframe[target_col].to_numpy()
            cols = [col for col in dataframe.columns if col != target_col]
        elif target_col is None:
            target_col = dataframe.columns[-1]
        
        if not all(item in dataframe.columns for item in cols) or target_col not in dataframe.columns or target_col in cols:
            raise ValueError('Invalid cols or target_col definition')

        self.targets = dataframe[target_col].to_numpy()
        self.features = dataframe[cols].to_numpy()

        self.model = None
        self.coef_ = None
        self.intercept_ = None
        self.mse = None
        self.residuals = None

    def fit(self):
        self.model = linear_model.LinearRegression().fit(self.features, self.targets)
        self.mse = mean_squared_error(self.targets, self.model.predict(self.features))
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.residuals = (self.targets - self.model.predict(self.features))

    def compute_p_values(self):

        if self.model is None:
            raise ValueError('Model needs to be fitted first')

        n, dim = self.features.shape

        X_with_const = np.hstack([self.features, np.ones((n, 1))])
        beta = np.append(self.coef_, self.intercept_, )
        mse = sum(self.residuals**2) / (n - (dim+1))
        var_beta = np.linalg.inv(X_with_const.T @ X_with_const).diagonal() * mse 

        # Calculate t-values
        t_values = beta / np.sqrt(var_beta)

        # Calculate p-values, two-sided
        p_values = [2 * (1 - t.cdf(np.abs(t_val), (n - (dim+1)))) for t_val in t_values]
        return p_values
    
    def compute_BIC(self):
        if self.model is None:
            raise ValueError('Model needs to be fitted first')
        n = len(self.targets)
        k = len(self.coef_)+1
        log_likelihood = -n/2*(np.log(2*np.pi)+np.log(self.mse)+1)
        return -2*log_likelihood + k*np.log(n)

