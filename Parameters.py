class ParameterSet():
    def __init__(self, M: int, N: int, T: int, R1: float, 
        R2: float, R3: float):
            self.M = M
            self.N = N
            self.T = T
            self.R1 = R1
            self.R2 = R2
            self.R3 = R3

    def __str__(self):
        return f"(M: {self.M}, N: {self.N}, T: {self.T}, \
R1: {self.R1}, R2: {self.R2}, R3: {self.R3})"