import random as rand
import pickle
from interface_valid_input import *

def random(num_bw,pattern_len):
    pattern = []
    for t in range(pattern_len):
        next = rand.randint(1,num_bw)
        pattern += [next]
    return pattern


def sawtooth(num_bw):
    pattern = []
    for t in range(num_bw):
        next = (t % num_bw) + 1
        pattern += [next]
    return pattern

def pulse(num_bw):
    pattern = []
    constant = rand.randint(1,num_bw)
    pulse = rand.randint(1,num_bw)
    for t in range(num_bw):
        if t % num_bw <= num_bw//2:
            next = pulse
        else:
            next = constant
        pattern += [next]
    return pattern

def auto_reg(num_bw,length):
    last_val = 1
    ret = []
    for t in range(length):
        if last_val % 2 == 0:
            next = last_val + t % num_bw
        else:
            next = last_val - t % num_bw
            
        if next >= num_bw:
            next = rand.choices([1, num_bw], weights=[0.1, 0.9])[0]
        elif next < 1:
            next = rand.choices([1, num_bw], weights=[0.9, 0.1])[0]

        last_val = next
        ret += [next]
    return ret

def mod_w_noise(num_bw,length):
    ret = []
    for t in range(length):
        next = (t % num_bw) + 1
        if t % num_bw == num_bw//2:
            next += (rand.randint(0,1) * 2) - 1
        ret += [next]
    return ret

def rand_spike(num_bw,length,spike):
    ret = []
    constant = rand.randint(1,num_bw)
    for t in range(length):
        if t % spike == 0:
            next = rand.randint(1,num_bw)
        else:
            next = constant
        ret += [next]
    return ret

#def custom(list):

class Policy():
    def __init__(self,policy_num,num_bw,length,pattern_type,pattern_len=0,spike=0):
        self.policy_num = policy_num
        self.list = []
        self.pattern = []
        if pattern_type == "random":
            self.pattern = random(num_bw,pattern_len)
        elif pattern_type == "sawtooth":
            self.pattern = sawtooth(num_bw)
        elif pattern_type == "pulse":
            self.pattern = pulse(num_bw)
        elif isinstance(pattern_type,list) :
            self.pattern = pattern_type

        if len(self.pattern) != 0:
            while len(self.list) < length-(length%len(self.pattern)):
                self.list += self.pattern
            for i in range((length%len(self.pattern))):
                self.list += [self.pattern[i]]

        else:
            if pattern_type == "auto reg":
                self.list = auto_reg(num_bw,length)
            elif pattern_type == "mod with noise":
                self.list = mod_w_noise(num_bw,length)
            elif pattern_type == "random spike":
                self.list = rand_spike(num_bw,length,spike)

    def get_bw(self,time):
        return self.list[time]

def pol_initialize(prompts=True):
    if prompts:
        statement = "Would you like to use the previous policy settings? (Y/N)"
        valid_inputs = ["Y","N","y","n"]
        print(statement)
        user_input = input()
        user_input = invalid_input(user_input,statement,valid_inputs)
    else: 
        user_input = "Y"
    ppfilename = 'policyparams.pk'
    #use previous policies
    if user_input == "Y" or user_input == "y":
        with open(ppfilename, 'rb') as fi:
            policy_params = pickle.load(fi)
            NUM_POLICIES = policy_params[0]
            NUM_BANDS = policy_params[1]
            POLICIES_LIST = policy_params[2]
            POLICIES = policy_params[3]
            LENGTH = policy_params[4]
    #create new policies
    elif user_input == "N" or user_input == "n":
        POLICIES_LIST = []
        POLICIES = []

        statement = "How many policies would you like to have?"
        valid_inputs = 'int'
        print(statement)
        user_input = input()
        user_input = invalid_input(user_input,statement,valid_inputs)
        NUM_POLICIES = int(user_input)

        statement = "How many bandwidths would you like to have?"
        print(statement)
        user_input = input()
        user_input = invalid_input(user_input,statement,valid_inputs)
        NUM_BANDS = int(user_input)

        statement = "How long would you like the game to be? (gamelength)"
        print(statement)
        user_input= input()
        user_input = invalid_input(user_input,statement,valid_inputs)
        LENGTH = int(user_input)

        for i in range(NUM_POLICIES):
            POLICY_NUM = i + 1
            print("What pattern type for Policy " + str(POLICY_NUM) + "? :")
            print("1 - random")
            print("2 - sawtooth")
            print("3 - pulse")
            print("4 - auto regressive")
            print("5 - mod with noise")
            print("6 - random spike")
            print("7 - custom")
            statement = "user_input (1-7): "
            valid_inputs = [1,2,3,4,5,6,7]
            print(statement)
            user_input = int(input())
            user_input = invalid_input(user_input,statement,valid_inputs)
            if user_input == 1:
                statement = "Insert repeated pattern length. (Must be smaller than gamelength)"
                valid_inputs = list(range(1,LENGTH+1))
                print(statement)
                user_input = int(input())
                user_input = invalid_input(user_input,statement,valid_inputs)
                PATTERN_LEN = user_input
                policy = Policy(POLICY_NUM,NUM_BANDS,LENGTH,"random",PATTERN_LEN)
            elif user_input == 2:
                policy = Policy(POLICY_NUM,NUM_BANDS,LENGTH,"sawtooth")
            elif user_input == 3:
                policy = Policy(POLICY_NUM,NUM_BANDS,LENGTH,"pulse")
            elif user_input == 4:
                policy = Policy(POLICY_NUM,NUM_BANDS,LENGTH,"auto reg")
            elif user_input == 5:
                policy = Policy(POLICY_NUM,NUM_BANDS,LENGTH,"mod with noise")
            elif user_input == 6:
                statement = "How often would you like spiking to occur? (Must be smaller then gamelength)"
                valid_inputs = list(range(1,LENGTH+1))
                print(statement)
                user_input = int(input())
                user_input = invalid_input(user_input,statement,valid_inputs)
                SPIKE = user_input
                policy = Policy(POLICY_NUM,NUM_BANDS,LENGTH,"random spike",0,SPIKE)
            else:
                print("Insert custom pattern separated by commas (no spaces)")
                string = input()
                convert = list(string.split(","))
                custom = []
                for i in range(len(convert)):
                    custom += [int(convert[i])]
                policy = Policy(POLICY_NUM,NUM_BANDS,LENGTH,custom)
            POLICIES_LIST += [policy.list]
            POLICIES += [policy]
        policy_params = [NUM_POLICIES,NUM_BANDS,POLICIES_LIST,POLICIES,LENGTH]        
        #save policies
        with open(ppfilename, 'wb') as fi:
            pickle.dump(policy_params,fi)
    return(NUM_POLICIES,NUM_BANDS,POLICIES_LIST,POLICIES,LENGTH)

