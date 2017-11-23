from .base_example import Example
import numpy as np

class Tuberculosis(Example):
    def _summaries():
        return [lambda x: x]


    def simulator(self, alpha, delta, tau):
        m = 20
        infected_hosts = np.array([[1]]) # list of haplotypes holding infectious hosts, we always start with one infected patient
        limit_exceeded = False
        round = 0

        while np.sum(infected_hosts) <= m and not limit_exceeded:
            round += 1
            # for each haplotype
            for cell in infected_hosts:
                # for each infectious host
                if cell[0] == 0:
                    continue
                for host in range(cell[0]):
                    # one of three things happen: transmission, mutation or recovery/death
                    chance = np.random.rand()
                    if chance < alpha:
                        if np.sum(infected_hosts) == m:
                            limit_exceeded = True
                            break
                        else:
                            cell += 1

                    chance = np.random.rand()
                    if chance < delta:
                        cell -= 1

                    chance = np.random.rand()
                    if chance < tau:
                        new_cell = [1]
                        cell -= 1
                        infected_hosts = np.vstack((infected_hosts, new_cell))

                if limit_exceeded:
                    break

        return sorted(infected_hosts[infected_hosts != 0], reverse=True)

tuberculosis = Tuberculosis()
