from .base_example import Example
import numpy as np


def nr_mutations(y):
    return y.shape[0]

def max_cluster(y):
    return y.max()

def nr_transmissions(y):
    return sum(np.where(y > 1, 1, 0))

class Tuberculosis(Example):

    def _summaries(self):
        return [nr_mutations, max_cluster, nr_transmissions]


    def simulator(self, alpha, delta, tau):
        m = 20
        infected_hosts = np.array([[1]]) # list of haplotypes holding infectious hosts, we always start with one infected patient
        limit_exceeded = False
        round = 0

        while np.sum(infected_hosts) <= m and not limit_exceeded:
            round += 1

            #reset if all died
            if np.sum(infected_hosts) == 0:
                infected_hosts = np.array([[1]])

            # for each haplotype
            for cell in infected_hosts:
                # for each infectious host
                if cell[0] == 0:
                    continue
                for host in range(cell[0]):
                    # one of three things happen: transmission, mutation or recovery/death
                    event = np.random.rand()
                    # if he dies -> no more action possible
                    if event < delta:
                        cell -= 1

                        continue

                    # otherwise, he can infect others or/and mutate
                    event = np.random.rand()
                    if event < alpha:
                        if np.sum(infected_hosts) == m:
                            limit_exceeded = True
                            break
                        else:
                            cell += 1


                    event = np.random.rand()
                    if event < tau and cell[0] > 1:
                        new_cell = [1]
                        cell -= 1
                        infected_hosts = np.vstack((infected_hosts, new_cell))



                if limit_exceeded:
                    break

        return np.array(sorted(infected_hosts[infected_hosts != 0], reverse=True))

tuberculosis = Tuberculosis()
