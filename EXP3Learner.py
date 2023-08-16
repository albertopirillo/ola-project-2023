import numpy as np
from Learner import Learner
import math
import pandas as pd
from numpy.random import choice


class EXP3Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.weights = np.ones(n_arms)
        self.estimated_reward = np.zeros(n_arms)
        self.gamma = 0.5
        self.probabilities = np.zeros(n_arms)

    def draw(self,w):
        choice = np.random.uniform(0, sum(w))
        choiceIndex = 0

        for weight in w:
            choice -= weight
            #print("choice:", choice)
            #print("weight: ", weight)
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1

    def pull_arm(self):
        weight_sum = float(sum(self.weights))
        for w in range(len(self.weights)):
            self.probabilities[w] = ((1.0 - self.gamma) * (self.weights[w] / weight_sum) + (self.gamma / self.n_arms))
        #print("probabilities: ", self.probabilities)
        index = self.draw(self.probabilities)
        return index

    def update(self, pulled_arm, reward):
        self.t += 1
        for a in range(self.n_arms):
            if a == pulled_arm:
                #print("aggiorno indice: ",a)
                self.estimated_reward[a] = reward / self.probabilities[a]
                self.weights[a] *= math.exp(self.estimated_reward[a] * self.gamma / self.n_arms)
            else:
                self.estimated_reward[a] = 0
        #print("weights", self.weights)
