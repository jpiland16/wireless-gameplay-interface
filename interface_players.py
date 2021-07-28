import pickle
from interface_valid_input import *
from interface_state import *
from RNNAd import *
import torch.optim as optim

def initialize_trans():
    pass

def initialize_Adversary(prompts,all_ad_params,all_runs,NUM_POLICIES,input_states,POLICIES):
    NUM_LAYERS = 2
    HIDDEN_DIM = 16
    LEARNING_RATE = 0.01
    LOOKBACK = 5
    Rnn_params = []
    ppfilename = 'Rnnparams.pk'
    if prompts:
        user_input = "N"
    else:
        user_input = "Y"
    #use previous neural network settings
    if user_input == "Y" or user_input == "y":
        with open(ppfilename, 'rb') as fi:
            Rnn_params = pickle.load(fi)
            NUM_LAYERS = Rnn_params[0]
            HIDDEN_DIM = Rnn_params[1]
            LEARNING_RATE = Rnn_params[2]
            LOOKBACK = Rnn_params[3]
    #set new neural network parameters
    elif user_input == "N" or user_input == "n":
        statement = "How many hidden layers would you like to use?"
        valid_inputs = 'int'
        print(statement)
        user_input = input()
        user_input = invalid_input(user_input,statement,valid_inputs)
        NUM_LAYERS = int(user_input)

        statement = "How many nodes would you like each hidden layer to have?"
        print(statement)
        user_input = input()
        user_input = invalid_input(user_input,statement,valid_inputs)
        HIDDEN_DIM = int(user_input)

        statement = "What would you like the learning rate to be? (decimal 0-1)"
        valid_inputs = 'float'
        print(statement)
        user_input = input()
        #update to specifically check that input is between 0-1
        user_input = invalid_input(user_input,statement,valid_inputs)
        LEARNING_RATE = float(user_input)

        statement = "How many timesteps in the past would you like the Adversary to consider?"
        valid_inputs = 'int'
        print(statement)
        user_input = input()
        user_input = invalid_input(user_input,statement,valid_inputs)
        LOOKBACK = int(user_input)

        Rnn_params = [NUM_LAYERS,HIDDEN_DIM,LEARNING_RATE,LOOKBACK]        
        #save neural network parameters
        with open(ppfilename, 'wb') as fi:
            pickle.dump(Rnn_params,fi)
    
    all_ad_params += [[NUM_LAYERS,HIDDEN_DIM,LEARNING_RATE,"RNN",LOOKBACK]]
    all_runs[2] = all_ad_params
    adversary = RNNAd(NUM_POLICIES, NUM_LAYERS, HIDDEN_DIM)    
    optimizer = optim.Adam(adversary.parameters(),lr=LEARNING_RATE)

    #create initial inputs
    next_input = []
    for i in range(NUM_POLICIES):
        input_states += [State(POLICIES[i])]
        next_input += [input_states[i].bw]
        next_input += [input_states[i].chosen]
    Ad_input = Tensor([[next_input]])
    return(all_ad_params,all_runs,adversary,optimizer,Ad_input)
