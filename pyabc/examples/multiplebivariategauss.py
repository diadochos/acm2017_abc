import numpy as np
from .base_example import Example


def group_means(x):
    return np.apply_along_axis(lambda x:np.mean(x,axis=0),1,x).flatten()

def distance(s1,s2):
    diff = s1 - s2 
    rmsd = np.mean(diff*diff)
    return np.sqrt(rmsd) 


class MultipleBivariateGauss(Example):

    #summary stat 
    def _summaries(self):
        return [group_means]

    def simulator(self,*mus):
        res = [None]*(int(len(mus) / 2))
        sigma = np.array([[0.01**2,0],[0.00,0.01**2]])

        for idx,i in enumerate(range(0,len(mus)-1,2)):
            res[idx] = np.random.multivariate_normal(np.array([mus[i], mus[i+1]]), sigma, 50)
        
        return np.array(res)


    def _distance(self):
        return distance

MultipleBivariateGauss = MultipleBivariateGauss()
