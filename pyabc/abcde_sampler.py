import time

import numpy as np
import scipy.stats as ss
import copy 

from .rejection_sampler import RejectionSampler
from .sampler import BaseSampler
from .utils import flatten_function

"""class doc"""


class ABCDESampler(BaseSampler):
   

    @property
    def weights(self):
        return self._weights

    @property
    def threshold(self):
        return self._threshold


    def __init__(self, priors, simulator, observation, summaries, distance='euclidean', verbosity=1, seed=None):
        # call BaseSampler __init__
        super().__init__(priors, simulator, observation, summaries, distance, verbosity, seed)

    def sample(self, nr_iter, nr_samples, nr_groups, distance='euclidean'):
        """Draw samples using Sequential Monte Carlo.

        Args:
            thresholds: list of acceptance threshold. len(thresholds defines number of SMC iterations)
            nr_particles: Number of particles used to represent the distribution
            distance: distance measure to compare summary statistics. (default) euclidean

        Returns:
            Nothing

        """
        if (nr_samples % nr_groups) != 0:
            raise ValueError("Nr samples has to be divided evenly into", nr_groups, 'groups')


        print("ABC-Differential-Evolution sampler started with number of samples: {}".format(nr_samples))

        self._reset()
        self._run_ABCDE_sampling(nr_samples, nr_groups, nr_iter)
        if self.verbosity == 1:
            print("Samples: %6d - Iterations: %10d - Acceptance rate: %4f - Time: %8.2f s" % (
                nr_samples, self.nr_iter, self.acceptance_rate, self.runtime))

    def calculate_fitness(self,curr_theta, delta, distance):
        error_distribution = ss.distributions.norm(0,delta)
        delta_prior = ss.distributions.expon(20).pdf(delta)
        fitness = self.priors.pdf(curr_theta)  * error_distribution.pdf(distance) * delta_prior
        return fitness

    def sample_non_matching_thetas(self,it,pool):     
        if self._pool_size == 1:
            return 0,0 

        theta_m = np.random.randint(0,self._pool_size )
        theta_n = np.random.randint(0,self._pool_size )

        while theta_n == theta_m: 
            theta_m = np.random.randint(0,self._pool_size ) 
            theta_n = np.random.randint(0,self._pool_size )

        return self._thetas[it-1,pool,theta_m,:], self._thetas[it-1,pool,theta_n,:] 


    def crossover(self,it,pool):
        for i in range(self._pool_size ):
            while True:
                #choose params we use for linear combination
                y1 = np.random.uniform(0.5,1)        
                y2 = np.random.uniform(0.5,1)
                b = np.random.uniform(-0.0001,0.0001)

                #get index for base particle
                idx_b = np.random.choice(np.arange(self._pool_size ), p = self._weights[it-1,pool,:])

                #get the 3 theta vectors used for crossover
                theta_t = self._thetas[it-1,pool,i,:]
                theta_b = self._thetas[it-1,pool,idx_b,:] #base_particle
                theta_m, theta_n = self.sample_non_matching_thetas(it,pool)

                #find a new theta, as a linear combination in the vector space of thetas within the cluster
                if self._sampling_mode == 'burnin':
                    theta_star = theta_t + y1*(theta_m - theta_n) + y2*(theta_b-theta_b) + b
                else:
                    theta_star = theta_t + y1*(theta_m - theta_n) +  b

                #keep some of the old features with probability (1-k)
                reset_probability = np.random.uniform(0,1, size = len(theta_star))

                for j in range(len(theta_star)):
                    if reset_probability[j] < (1-self.k):
                        theta_star[j] = theta_t[j]

                #make sure we found a great theta that works with our prior and then we can simulate and see how well it fits the data
                if self.priors.pdf(theta_star) > 0:

                    Y = self.simulator(*(np.atleast_1d(theta_star)))  # unpack thetas as single arguments for simulator
                    stats_y = flatten_function(self.summaries, Y)
                    d = self.distance(self._stats_x, stats_y)
                    
                    #TODO save the deltas(?)
                    delta = np.random.exponential(20)

                    #calculate fitness values for both
                    #TODO how to choose theta, can we just reuse the old weight(?)
                    proposal_fitness = self.calculate_fitness(theta_star, delta, d)
                    previous_fitness = self.calculate_fitness(theta_t, delta, self.distances[it-1,pool,i])

                    MH_prob = min(1, proposal_fitness / previous_fitness)
                    u = np.random.uniform(0,1)
                    
                    if u < MH_prob:
                        self._thetas[it,pool,i,:] = theta_star
                        self._distances[it,pool,i] = d
                        self._weights[it,pool,i] = proposal_fitness
                    else:
                        self._thetas[it,pool,i,:] = theta_t 
                        self._distances[it,pool,i] = distances[it-1,pool,i]
                        self._weights[it,pool,i] = previous_fitness

                    break
        return 
            
    def mutate(self,it,pool):
        #TODO: is this really the best method to choose the step_size(?)
        self._sigmas[it,pool,:] = 2 * np.cov(self._thetas[it-1, pool, : , :].T, aweights=self._weights[it-1,pool,:])

        for i in range(self._pool_size):
            while True:
                #TODO figure out if we need this step_size, or we can calculate spme variance based on particles as in SMC?
                theta_star = np.atleast_1d(ss.multivariate_normal(self._thetas[it-1,pool,:], sigma, allow_singular=True).rvs()) 
                theta_old  = self._thetas[it-1,pool,i,:]

                #make sure we found a great theta that works with our prior and then we can simulate and see how well it fits the data
                if self.priors.pdf(theta_star) > 0:
                    Y = self.simulator(*(np.atleast_1d(theta_star)))  # unpack thetas as single arguments for simulator
                    stats_y = flatten_function(self,summaries, Y)
                    d = self.distance(self._stats_x, stats_y)

                    #only change the delta parameter during burnin
                    #TODO how to estimate posterior distribution of delta(?) 

                    if self.sampling_mode == 'burnin':
                        delta = np.random.exponential(20)
                    else: 
                        delta = self.delta 

                    proposal_fitness = self.calculate_fitness(theta_star, delta, d)
                    previous_fitness = self.calculate_fitness(theta_old, delta, self.distances[it-1,pool,i])

                    #calculate MH probability
                    MH_prob = min(1,proposal_fitness / previous_fitness)

                    u = np.random.uniform(0,1)

                    if u < MH_prob:
                        self._thetas[it,pool,i,:] = theta_star
                        self._distances[it,pool,i] = d
                        self._weights[it,pool,i] = proposal_fitness

                    else:
                        self._thetas[it,pool,i,:] = theta_t 
                        self._distances[it,pool,i] = distances[it-1,pool,i - 1]
                        self._weights[it,pool,i] = previous_fitness

                    break

        return 


    def migrate(self,it):

        K = np.random.randint(1,self._nr_groups)
        groups = np.arange(self._nr_groups)
        np.random.shuffle(groups)
        groups = groups[0:K]
        
        #setup arrays to temporarily store the indices of the particles we want to swap
        weak_particles_idx = [None] * K

        #first choose all the weak_particles by their inverse of their weights and store their index
        for i in range(K):
            curr_group = groups[i]
            #normalize thee group weights
            #TODO do this somewhere else 
            group_weights = 1 / (self.weights[it-1,curr_group,:]+1) #dont divide by zero so add bias(?)
            group_weights = group_weights  / sum(group_weights)
            print(group_weights)
            weak_particles_idx[i] = np.random.choice(np.arange(0,self._pool_size), p = group_weights)


        #close the cycle by setting the first particles to the last
        previous_theta = self.thetas[it-1,groups[0],weak_particles_idx[0]] 
        previous_weight = self.weights[it-1,groups[0],weak_particles_idx[0]] 

        self.thetas[it-1,groups[0],weak_particles_idx[0]] = thetas[groups[K-1],weak_particles_idx[K-1]] 
        self.weights[it-1,groups[0],weak_particles_idx[0]] = weights[groups[K-1],weak_particles_idx[K-1]] 

        for i in np.arange(1,K):
            curr_group = groups[i]
            
            #temporarily store our particles before we overwrite them with the previous 
            temp_theta = self.thetas[it-1,curr_group,weak_particles_idx[i]]
            temp_weight =  self.weights[it-1,curr_group,weak_particles_idx[i]]

            self,thetas[it-1,curr_group,weak_particles_idx[i]] = previous_theta
            self.weights[it-1,curr_group,weak_particles_idx[i]] = previous_weight

            previous_theta = temp_theta 
            previous_weight = temp_weight

        return


    def init_thetas(self,t): 
        for i in range(self._nr_groups):
            self._thetas[t,i,:,:] = self.priors.sample(self._pool_size)
            for j in range(self._pool_size):
                curr_theta = self._thetas[t,i,j,:]
                Y = self.simulator(*(np.atleast_1d(curr_theta)))  # unpack thetas as single arguments for simulator
                stats_y = flatten_function(self.summaries, Y)
                d = self.distance(self._stats_x, stats_y)
                #TODO which delta to use(?)
                self._weights[t,i,j] = self.calculate_fitness(curr_theta, 1, d)
                self._distances[t,i,j] = d 
            #normalize weights within pool
            self._weights[t,i,:] = self._weights[t,i,:] / sum(self._weights[t,i,:])


    def _run_ABCDE_sampling(self, nr_samples, nr_groups, nr_iterations):
        X = self.observation
        self._stats_x = flatten_function(self.summaries, X)
        num_priors = len(self.priors) 
        T = nr_iterations

        #TODO, how do we want to calculate our iterations(?)
        nr_iter = 0

        #TODO make these decision tuning parameters arguments of the sampler 
        #In practice, the ‘‘decision’’ tuning parameters will be small 
        #(e.g., α = β = 0.10), such that a majority of the particle updates are done in the DE Crossover Step
        alpha = 0.1
        beta = 0.1 
        self.k = 0.9 

        self._nr_groups = nr_groups
        self._pool_size = int(nr_samples / nr_groups)
        self._sampling_mode = 'burnin'        

        # create a large array to store all particles (THIS CAN BE VERY MEMORY INTENSIVE) - work with current/previous if too hard on memory
        # make this a class variable, so we can perform all operations in place and dont have to pass new arguments
        self._thetas = np.zeros((T,self._nr_groups, self._pool_size, num_priors))
        self._weights = np.zeros((T,self._nr_groups, self._pool_size))
        self._sigmas = np.zeros((T, self._nr_groups, self._pool_size))
        self._distances = np.zeros((T,self._nr_groups, self._pool_size))

        #set this here 
        self._nr_iter = 0

        start = time.clock()

        for t in range(nr_iterations):
            # init particle pools by simply sampling from the prior for each pool
            if t == 0:
                if self.verbosity:
                    print('initializing pools')
                self.init_thetas(t)

            else:             
                if self.verbosity:
                    print('starting iteration[', t, ']')

                #TODO make this a parameter of the sampler 
                if t > 100:
                    self._sampling_mode = 'sample'
                    self._delta = 0.3

                p1 = np.random.uniform(0,1)

                if p1 < alpha:
                    #Migrate between the groups such as to diversify them, this happens by choosing n groups and cycling the weakest particle to the neighbouring group
                    self.migrate(t) 

                #for each pool of particles do the following:
                for i in range(self._nr_groups):
                    p2 = np.random.uniform(0,1)   
                    if p2 < beta: 
                        #This slightly perturbs each particle within a group using a perturbation kernel (mv-gaussian)
                        self.mutate(t,i)
                    else:
                        #This generates new particles, by operating in the vector space of thetas for a particular particle cluster
                        self.crossover(t,i)


        self._runtime = time.clock() - start
        self._nr_iter = nr_iter
        self._acceptance_rate = nr_samples / self.nr_iter
        self._weights = weights
        self._Thetas = thetas[T - 1, :, :]
        self._distances = distances

        return thetas[T - 1, :, :]

    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)
