import numpy as np

class CUSUM:
    def __init__(self, M, eps, h):
        self.M = M          #M: number of first valid samples
        self.eps = eps      #eps: exploration quantity
        self.h = h          #h: threshold for change detection
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return 0  #return 0 in no changes are detected
        else:
            s_plus = (sample - self.reference) - self.eps       #positive deviation from the reference point at time t
            s_minus = -(sample - self.reference) - self.eps     #negative deviation from the reference point at time t
            self.g_plus = max(0, self.g_plus + s_plus)          #cumulative positive deviation
            self.g_minus = max(0, self.g_plus + s_minus)        #cumulative negative deviation
            return self.g_plus > self.h or self.g_minus > self.h #return 1 if changes are detected

    #method to reset variables in case a change is found    
    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0
        self.reference = 0