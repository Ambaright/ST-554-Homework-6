# Author: Amanda Baright
# Date: 03.11.2026
# Purpose: ST 554 Homework 6, Create a Python Class

# This Python file will be used to create a class called `SLR_slope_simulator` that will be used to encapsulate
# the simulation of the sampling distribution.

# We first want to load in some needed libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn import linear_model

# We now define our class
class SLR_slope_simulator:
    # We want to initialize the class using __init__ with arguments self, beta_0, beta_1, x, sigma, and seed
    # We will also create initial attributes of beta_0, beta_1, sigma, x, n, rng, and slopes (an empty list)
    def __init__(self, beta_0, beta_1, x, sigma, seed):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.x = x
        self.n = len(x) # length of x
        self.rng = default_rng(seed) # random seed
        self.sigma = sigma
        self.slopes = [] # an empty list
        
    # Now we get into a few required methods that will work on an already created instance of the class
    
    # The first one will be the generate_data method that will generate one dataset, 
    #returning x and y as either a tuple or a set of arrays containing these x and y values
    def generate_data(self):
        """
        Generate a dataset modeled from the line plus some random normal deviation.
        Here our line is: Y = beta_0 + beta_1*x + epsilon, where epsilon ~ N(0, sigma^2)
        """
        y = self.beta_0 + self.beta_1 * self.x + self.rng.normal(0, self.sigma, self.n)
        return self.x, y
    
    # Next, we want to create the fit_slope method that will take in an x and y and fit the SLR model,
    # returning the estimated slope as a float
    def fit_slope(self, x, y):
        """
        Fit a SLR model with the provided x and y and return the estimated slope.
        """
        # prepare to fit a Linear Regression
        reg = linear_model.LinearRegression()
        
        # reshape x for sklearn
        fit = reg.fit(x.reshape(-1,1), y)
        
        # return the slope coefficient
        return fit.coef_[0]
    
    # Next, we want to create the run_simulations method that takes in a number of simulations argument (num_simulations) and uses the
    # generate_data() and fit_slope() methods within a for loop. This will not return anything, but will simply
    # modify the slopes attribute and will replace the empty list with an array of slope estimates.
    def run_simulations(self, num_simulations):
        """
        A Method to run the simulations with the generate_data() and fit_slope() methods within a for loop.
        Takes in a num_simulations argument that specifies the number of simulations.
        Nothing will be returned, instead the slopes attribute will be modified to have an array of slope estimates.
        """
        # create a temporary empty list for the slopes
        temp_slopes = []
        for _ in range(num_simulations):
            x_sim, y_sim = self.generate_data() # use generate_data to get data for x and y
            slope_est = self.fit_slope(x_sim, y_sim) # use fit_slope to fit the SLR model
            temp_slopes.append(slope_est) # append new slope estimate to temporary list
            
        # now modify the slopes attribute to have an array of estimated slopes from temp_slopes
        self.slopes = np.array(temp_slopes)
        
    # Next, we want to be able to plot our distribution by creating plot_sampling_distribution method
    # This will check if the slopes attribute has length greater than 0 
    # (if it doesn't we print a message that run_simulations() must be called first).
    # If it is greater than length 0, it should produce a histogram of the slopes approximating the sampling distribution.
    def plot_sampling_distribution(self):
        """
        This method will first check to see that the slopes attribute has a length greater than 0.
        If the length is not greater than 0, we print a message to prompt the user to call run_simulations() method first.
        If the length is greater than 0, we than plot a histogram of the slopes approximating the sampling distribution.
        """
        # check the length
        if len(self.slopes) == 0:
            print("Error: run_simulations() method must be called first.")
        else:
            # plot the histogram
            plt.hist(self.slopes)
            plt.title("Sampling Distribution of the Slope Estimator")
            plt.xlabel("Estimated Slope")
            plt.ylabel("Frequency")
            plt.show()
            
    # Finally, we want to create a find_prob method that will take in a value and a sided arguement.
    # It will check on the slopes attribute above. If the length is bigger than 0, it should approximate the probability of being
    # "above", "below" or "two-sided" (values for sided)
    def find_prob(self, value, sided):
        """
        Method to approximate the probability based on the simulated distribution.
        Value is taken as the value being compared for the type of probability being computed.
        Sided is the type of probability approximation.
            - Above: approximate the probability of being larger than the provided value.
            - Below: approximate the probability of being smaller than the provided value.
            - Two-sided: check that the value is above or below the median.
                If above, you find two times the probability of being larger.
                If below, you find two times the probability of being smaller.
        Returns the probability, between 0 and 1.
        """
        
        # First we want to check the length of the slopes attribute. Returns None if length is zero.
        if len(self.slopes) == 0:
            print("Error: run_simulations() method must be called first.")
            return None
        
        # Check if sided = above
        if sided == "above":
            # Probability of being larger than the value
            return (self.slopes > value).mean()
        
        # Check if sided = below
        elif sided == "below":
            # Probability of being smaller than the value
            return (self.slopes < value).mean()
        
        # Check if sided = two-sided
        elif sided == "two-sided":
            # Get the median
            median_val = np.median(self.slopes)
            
            # Checks if the value is above or below the median
            if (value > median_val):
                # need to return two times the probability of being larger
                return 2 * (self.slopes > value).mean()
            else:
                # need to return two times the probability of being smaller
                return 2 * (self.slopes < value).mean()
            
# This concludes our class creation. Now we will use the class and its method to add some code to run.

# ---------------------------------------------------------------------------------------------------

# Execution Section

# ---------------------------------------------------------------------------------------------------

# We want to create an instance of the object with 
# beta_0 = 12, beta_1 = 2, x = np.array(list(np.linspace(start = 0, stop = 10, num = 11))*3), sigma = 1, seed = 10

sim = SLR_slope_simulator(beta_0 = 12, beta_1 = 2, x = np.array(list(np.linspace(start = 0, stop = 10, num = 11))*3), sigma = 1, seed = 10)
                          
# Now we will call out plot_sampling_distribution() method, which should produce an error
sim.plot_sampling_distribution()

# Now we want to run 10000 simulations
sim.run_simulations(10000)

# Now we plot the distribution
sim.plot_sampling_distribution()
                          
# Now we can approximate the two-sided probability of being larger than 2.1
prob = sim.find_prob(2.1, "two-sided")
print(f"Two-sided probability of being larger than 2.1 is: {prob}")
                          
# Finally, we print out the values of the simulated slopes using the attribute
print("Simulated slopes list: ", sim.slopes)
        
        
        
        