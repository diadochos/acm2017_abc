import numpy as np
import copy
import scipy.stats as ss
from .base_example import Example

#simulate the study of a word
def study_word(word, memory_trace, u , g ,c,w):
    studied_word = copy.deepcopy(memory_trace)
    for i in range(w):
        if (memory_trace[i] == 0):
            u_copy = np.random.uniform(0,1)
            c_copy = np.random.uniform(0,1)
            if u_copy < u:
                if c_copy < c:
                    studied_word[i] = word[i]
                else:
                    studied_word[i]  = np.random.geometric(g)
            else:
                studied_word[i] = studied_word[i]
    return studied_word

#get matching features
def get_nonzero_feature_matches(probe, trace):
    non_zero_matching = np.equal(probe,trace) * (trace > 0)
    non_zero_mismatching = np.not_equal(probe,trace) * (trace > 0)
    njq = np.sum(non_zero_mismatching)
    njm = np.sum(non_zero_matching)

    #which tells us the number of non-zero matching type features in the j-th trace with the value of i
    nijm = np.zeros(max(probe)+1)
    for i in np.arange(1,len(nijm)):
        nijm[i] = np.sum(non_zero_matching * np.equal(probe,i))

    return njq, njm, nijm

#calculate similarity between probe and trace
def calculate_similarity(probe, trace,u,g,c):
    njq,njm, nijm = get_nonzero_feature_matches(probe,trace)
    prod = 1
    for i in range(max(probe)+1):
        if nijm[i] > 0:
            geometrical = g*pow((1-g),i-1)
            prob_ratio =  pow((c + (1-c)*geometrical) / geometrical, nijm[i])
            prod *= prob_ratio

    lambda_j = pow((1-c),njq) * prod
    return lambda_j

def study_and_test(study_list, test_list,u,g,c,w):
    n_study = len(study_list)
    n_test = len(test_list)
    #study phase
    episodic_matrix = np.zeros(study_list.shape, dtype = int)
    for i in range(n_study):
        episodic_matrix[i,:] = study_word(study_list[i],episodic_matrix[i],u,g,c,w)

    #test phase
    results = np.zeros(n_test)
    for i in range(n_test):
        total = 0
        for j in range(n_study):
            total += calculate_similarity(test_list[i],episodic_matrix[j],u,g,c)

        phi = 1/(n_test) * total

        if phi > 1:
            results[i] = 1
        else:
            results[i] = 0

    return results

def sim(u,g,c,w,conditions):
    study_lists = [None] * len(conditions)
    test_lists =  [None] * len(conditions)

    #initialize study and test material
    for i in range(len(conditions)):
        full_list = np.random.geometric(g,(conditions[i]*2,w))
        study_lists[i] = full_list[0:conditions[i]]
        test_lists[i] = full_list

    #Now study and test for each condition
    y = [None] * len(study_lists)
    for i in range(len(study_lists)):
        y[i] = study_and_test(study_lists[i], test_lists[i],u,g,c,w)

    return y

def calculate_hit_fa_rate(results):
    summarized = np.zeros((len(results), 4))
    for i in range(len(results)):
        result = results[i]

        #currently only supports test lists that are twice the size of study
        n_study = int(len(result) / 2)
        n_test = len(result)

        n_hits = np.sum(result[0:n_study])
        n_fa = np.sum(result[n_study:])

        p_hit = n_hits / n_study
        p_false = n_fa / (n_test - n_study)

        #print('hit_rate:', p_hit)
        #print('false_alarm_rate:', p_false)
        #print(n_fa)

        YjHIT = ss.distributions.binom(n_study,p_hit).pmf(n_hits)
        YjFA = ss.distributions.binom((n_test - n_study),p_false).pmf(n_fa)

        summarized[i,:] = [n_hits,n_fa, n_study, (n_test - n_study)]

    return summarized

class REMMemory(Example):
    def simulator(self, u, c, g, w=9, conditions=[10]):
        return sim(u,g,c,w,conditions)

    def _summaries(self):
        return [calculate_hit_fa_rate]

rem_memory = REMMemory()
