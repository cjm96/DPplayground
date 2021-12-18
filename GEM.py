import numpy as np
from scipy.stats import beta, uniform
from scipy.special import betaincinv

def stick_breaking(alpha, K=100, N=1):
    """
    The simplest, naive implementaion of the stick breaking construction
    WARNING - this is very slow! You probably want to use GEM instead
    
    INPUTS
    ------
    alpha: float
        the DP concentration parameter
    K: int
        the maximum number of clusters
    N: int
        the number of sticks to break (i.e. how many realisations)
        
    RETURNS
    -------
    pi: array shaped (K, N)
        the DP mixture weights
    """
    # initialise an empty array for the weights
    pi = np.zeros((K, N))
    
    # loop over the realisations
    for j in range(N):
        b = beta(1, alpha).rvs(K)
        
        # loop over the clusters
        for i in range(K):
            
            # compute the weight \pi_{ij}
            if i==0:
                pi[i,j] = b[i]
            else:
                vec = [1-b[kp] for kp in np.arange(i)]
                pi[i,j] = b[i] * np.prod(vec)
                
    #return weights
    return pi

def beta1a_rvs(a, K=1):
    """
    Function to draw random variable from several different beta distributions simultaneously.
    
    \beta_k ~ Beta(1, \alpha), for k=1,2,...,K.
    
    Shape parameter alpha can be an array with several values descibing different distributions.
    
    INPUTS
    ------
    a: np.array, shape (N,)
        vector of shape parameters describing several different beta distributions
    K: int
        the maximum number of clusters
        
    RETURNS
    -------
    beta: np.array, shape (K, N)
        the output random variables
    """
    A = np.asarray(a)
    assert len(np.asarray(A).shape)==1, "input array a must be of shape (N,)"
    return betaincinv(np.ones_like(A), A, uniform().rvs(size=(K, len(A))))
 
def GEM(alpha, K=100, N=1):
    """
    The stick-breaking procedure for the weights \pi~GEM(\alpha)
    (GEM stands for Griffiths, Engen and McCloskey)
    
    INPUTS
    ------
    alpha: float OR array shaped (N,)
        the DP concentration parameter
        - if float, then alpha describes a single DP and this function draws 
        N realisations from the size breaking procedure
        - if alpha is an array then it describes a mixture of DPs and  this 
        function draws a single realisation from each
    K: int
        the maximum number of clusters
    N: int
        the number of sticks to break (i.e. how many realisations)
        (this is used only if alpha is a float)
    
    RETURNS
    -------
    pi: array shaped (K, N)
        the DP mixture weights
    """
    if isinstance(alpha, float) or isinstance(alpha, int):
        b = beta1a_rvs(alpha*np.ones(N), K=K).T
    
    elif isinstance(alpha, np.ndarray):
        b = beta1a_rvs(alpha, K=K).T
    
    else:
        raise TypeError('input alpha must be a float or array shaped (N,)')
            
    pi = b.copy()
    pi[...,1:] = b[...,1:] * (1 - b[...,:-1]).cumprod(axis=-1)
    return pi.T
    


if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt

    alpha = 20
    K, N = 100, 100

    start = time.time()
    pi = stick_breaking(alpha, K=K, N=N)
    end = time.time()
    print(f"stick_breaking: very slow,{end-start: .2e}s")
    plt.plot(np.cumsum(pi, axis=0))
    plt.xlabel("k"); plt.ylabel(r"$\sum_{i=0}^{k-1}\pi_i$"); plt.show()

    start = time.time()
    pi = GEM(alpha, K=K, N=N)
    end = time.time()
    print(f"GEM (with single alpha): fast,{end-start: .2e}s")
    plt.plot(np.cumsum(pi, axis=0))
    plt.xlabel("k"); plt.ylabel(r"$\sum_{i=0}^{k-1}\pi_i$"); plt.show()

    start = time.time()
    pi = GEM(np.ones(N)*alpha, K=K)
    end = time.time()
    print(f"GEM (with vector of alphas): fast,{end-start: .2e}s")
    plt.plot(np.cumsum(pi, axis=0))
    plt.xlabel("k"); plt.ylabel(r"$\sum_{i=0}^{k-1}\pi_i$"); plt.show()
