class State():
    def __init__(self,policy,time=0,chosen=2):
        self.policy = policy
        self.time = time
        self.bw = self.policy.get_bw(self.time)
        self.chosen = chosen
        self.state = [self.policy.policy_num,self.time,self.bw,self.chosen]
    def get_next(self):
        self.time += 1
        self.bw = self.policy.get_bw(self.time)
        self.chosen = 2
        self.state = [self.policy.policy_num,self.time,self.bw,self.chosen]
        return self