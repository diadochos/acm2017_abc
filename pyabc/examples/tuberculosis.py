from .base_example import Example
import numpy as np

SAMPLE_SIZE = 20

def T1(y):
    return y[y>0].shape[0] / SAMPLE_SIZE

def T2(y):
    """genetic diversity"""
    return 1 - np.sum(np.power(y/SAMPLE_SIZE, 2))

class Tuberculosis(Example):

    def _summaries(self):
        return [T1, T2]


    def simulator(self, alpha, tau=0.198, delta=0):
        infected_hosts = np.array([[1]]) # list of haplotypes holding infectious hosts, we always start with one infected patient
        limit_exceeded = False
        round = 0

        while np.sum(infected_hosts) <= SAMPLE_SIZE and not limit_exceeded:
            round += 1

            #reset if all died
            if np.sum(infected_hosts) == 0:
                infected_hosts = np.array([[1]])

            # new event happens
            # choose which genotype is affected
            k = np.random.choice(range(infected_hosts.shape[0]), p=(infected_hosts/np.sum(infected_hosts)).flatten())
            cell = infected_hosts[k]

            # which event?
            event = np.random.choice(range(3), p=np.array([alpha, delta, tau]) / np.sum(np.array([alpha, delta, tau])))

            # one of three things happen: transmission, mutation or recovery/death
            if event ==  0:
                if np.sum(infected_hosts) == SAMPLE_SIZE:
                    limit_exceeded = True
                    break
                else:
                    cell += 1

            elif event == 1:
                cell -= 1

            else:
                new_cell = [1]
                cell -= 1
                infected_hosts = np.vstack((infected_hosts, new_cell))


            if limit_exceeded:
                break

        for i in range(len(infected_hosts), SAMPLE_SIZE):
            infected_hosts = np.vstack((infected_hosts, [0]))

        return np.array(sorted(infected_hosts, reverse=True))[:SAMPLE_SIZE].flatten()

tuberculosis = Tuberculosis()
