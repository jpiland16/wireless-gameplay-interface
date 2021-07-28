import pickle
import math
from interface_policies import *
from interface_state import *
from RNNAd import *
from interface_visuals import *
from interface_valid_input import *
from interface_players import *
from interface_accuracy import *
from interface_saves import *
from torch import nn, Tensor
import torch.optim as optim
from tqdm import tqdm

# Legacy-ish
from Transmitters import *
from GameParameters import *

reset_saves()

ppfilename = 'all_interface_runs.pk'
with open(ppfilename, 'rb') as fi:
    all_runs = pickle.load(fi)
    all_pol_params = all_runs[0]
    all_rewards = all_runs[1]
    all_ad_params = all_runs[2]
    all_acc = all_runs[3]
    all_trans_choices = all_runs[4]
    all_ad_choices = all_runs[5]
    all_scores = all_runs[6]
    all_notes = all_runs[7]

user_input = ''
POLICIES = []
#POLICIES = list of Policy objects
POLICIES_LIST = []
#POLICIES_LIST = list of bandwidth choices for each policy
NUM_POLICIES = 0
NUM_BANDS = 0
LENGTH = 0

print("Welcome to THE GAME")
#generate policies
NUM_POLICIES,NUM_BANDS,POLICIES_LIST,POLICIES,LENGTH = pol_initialize()
all_pol_params += [[NUM_POLICIES,NUM_BANDS,POLICIES_LIST,POLICIES,LENGTH]]
all_runs[0] = all_pol_params

#set up reward values
statement = "Would you like to use previous reward values?"
valid_inputs = ["Y","y","N","n"]
print(statement)
user_input = input()
user_input = invalid_input(user_input,statement,valid_inputs)
if user_input == "Y" or user_input == "y":
    #Team A successful transmission
    R1 = all_rewards[0]
    #Team A policy switch penalty
    R2 = all_rewards[1]
    #Team B successful jam; -R3*log base 2(N) for Team A communication
    R3 = all_rewards[2]
elif user_input == "N" or user_input == "n":
    statement = "What would you like Team A successful transmission reward to be (integer)?"
    valid_inputs = 'int'
    print(statement)
    user_input = input()
    user_input = invalid_input(user_input,statement,valid_inputs)
    R1 = int(user_input)
    
    statement = "What would you like Team A policy switch penalty to be (positive integer)?"
    valid_inputs = 'int'
    print(statement)
    user_input = input()
    user_input = invalid_input(user_input,statement,valid_inputs)
    R2 = int(user_input)
    
    statement = "What would you like Team B (Adversary) successful jam reward to be (integer)?"
    valid_inputs = 'int'
    print(statement)
    user_input = input()
    user_input = invalid_input(user_input,statement,valid_inputs)
    R3 = int(user_input)
all_rewards = [R1,R2,R3]
all_runs[1] = all_rewards

# ---- Jonathan
next_input = [0] * (NUM_POLICIES * 2)
past_inputs = []
# ---- /Jonathan

Ad_input = []
#Ad_input = list/Tensor of next possible bandwidth for each policy
input_states = []
#input_states = list of State objects
time = 0
end = False
ad_correct_pol = 0
ad_correct_bw = 0
trans_correct = 0
trans_pols = []
trans_bws = []
ad_pols = []
ad_bws = []
all_ad_acc_pol = []
all_ad_acc_bw = []
all_trans_acc = []
adversary_score = 0
transmitter_score = 0

#Set up Adversary NN 
statement = "Would you like to use previous Adversary Neural Network settings?"
valid_inputs = ["Y","y","N","n"]
print(statement)
user_input = input()
user_input = invalid_input(user_input,statement,valid_inputs)
if user_input == "Y" or user_input == "y":
    all_ad_params,all_runs,adversary,optimizer,Ad_input = initialize_Adversary(False,all_ad_params,all_runs,NUM_POLICIES,input_states,POLICIES)
elif user_input =="N" or user_input =="n":
    all_ad_params,all_runs,adversary,optimizer,Ad_input = initialize_Adversary(True,all_ad_params,all_runs,NUM_POLICIES,input_states,POLICIES)

transmitter = RandomTransmitter(NUM_POLICIES)
last_transmitter_policy_index = transmitter.start_policy
game_state = GameState(
    GameParameterSet(-1, NUM_POLICIES, -1, -1, -1, -1), [], [])

LOOKBACK = all_ad_params[-1][4]
ad_choices = []

for i in range(LENGTH):
    #Adversary prediction
    output,ad_pol = adversary.predict(Ad_input)

    #Transmitter prediction
    new_transmission_policy_index, _ = transmitter.get_policy(game_state)

    transmitter_switched_policy = (new_transmission_policy_index >= 0 and 
        new_transmission_policy_index != last_transmitter_policy_index)
        
    trans_pol = (new_transmission_policy_index if transmitter_switched_policy
        else last_transmitter_policy_index)

    last_transmitter_policy_index = trans_pol
    trans_pol += 1
    trans_pols += [trans_pol]

    #Adversary loss/RL
    trans_bw,trans_vec,past_trans_vecs,ad_pol,ad_choices,ad_bw = Ad_learn(trans_pol,POLICIES,time,NUM_POLICIES,input_states,past_trans_vecs,LOOKBACK,output,optimizer,ad_choices,ad_pol,Ad_input)
    #Tranmsitter loss/RL/Q val updates
    # -- Trans_learn() --
    #accuracy + chart progress
    ad_correct_pol,ad_correct_bw,ad_acc_pol,ad_acc_bw = Ad_accuracy(trans_pol,ad_pol,ad_correct_pol,trans_bw,ad_bw,ad_correct_bw,time)

    # ---- Jonathan added this
    all_ad_acc_bw += [ad_acc_pol]
    all_ad_acc_pol += [ad_acc_pol]
    trans_bws += [trans_bw]
    ad_bws += [ad_bw]
    # ---- /end Jonathan

    trans_correct,trans_acc = Trans_accuracy(trans_bw,ad_bw,trans_correct,time)
    
    # ---- Jonathan
    all_trans_acc += [trans_acc]
    ad_pols = ad_choices
    # ---- /end Jonathan

    # Calculate the rewards
    
    intercepted = (ad_bw == trans_bw)
    transmitter_communicated_switch = True
    
    if intercepted:
        adversary_score += R3
    else:
        # TODO: Don't assume the receiver is correct here
        transmitter_score += R1
    
    if transmitter_switched_policy:
        transmitter_score -= R2
    
    if transmitter_switched_policy and transmitter_communicated_switch:
        # TODO: uncomment the next line if using Receiver
        # transmitter_score -= R3 * math.log2(NUM_POLICIES)
        pass

    game_progress(trans_pols,ad_choices,POLICIES_LIST,time,all_ad_acc_pol,all_ad_acc_bw,all_trans_acc,transmitter_score,adversary_score)
    #set up for next timestep
    if i != LENGTH-1:
        time,input_states,next_input,past_inputs,Ad_input = Ad_continue(time,NUM_POLICIES,input_states,trans_pol,next_input,past_inputs,LOOKBACK)
        # -- Trans_continue() --

run_num = len(all_pol_params)
graph_choices(trans_pols,ad_pols,trans_bws,ad_bws,LENGTH,run_num)

scores = [transmitter_score,adversary_score]
all_scores += [scores]
all_runs[6] = all_scores

note = ""
statement = "Would you like to add a note to save with this game? (Y/N)"
valid_inputs = ["Y","y","N","n"]
print(statement)
user_input = input()
user_input = invalid_input(user_input,statement,valid_inputs)
if user_input == "Y" or user_input == "y":
    statement = "Enter your short note (max 40 characters): "
    print(statement)
    user_input = input()
    note = invalid_input(user_input,statement,"note")
all_notes += [note]
all_runs[7] = all_notes
save(all_runs,all_ad_acc_pol,all_ad_acc_bw,all_trans_acc,all_acc,trans_pols,trans_bws,all_trans_choices,ad_pols,ad_bws,all_ad_choices)