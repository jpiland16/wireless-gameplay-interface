import pickle
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

#reset_saves()

ppfilename = 'all_interface_runs.pk'
with open(ppfilename, 'rb') as fi:
    all_runs = pickle.load(fi)
    all_pol_params = all_runs[0]
    all_ad_params = all_runs[1]
    all_acc = all_runs[2]
    all_trans_choices = all_runs[3]
    all_ad_choices = all_runs[4]
    all_notes = all_runs[5]

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

#Set up Adversary NN 
statement = "Would you like to use previous Adversary Neural Network settings?"
valid_inputs = ["Y","y","N","n"]
print(statement)
user_input = input()
user_input = invalid_input(user_input,statement,valid_inputs)
if user_input == "Y" or user_input == "y":
    all_ad_params,all_runs,adversary,optimizer,Ad_input = initialize_Adversary(False)
elif user_input =="N" or user_input =="n":
    all_ad_params,all_runs,adversary,optimizer,Ad_input = initialize_Adversary(True)
all_runs[1] = all_ad_params

LOOKBACK = all_ad_params[4]
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

transmitter = ExampleTransmitter(NUM_POLICIES)
last_transmitter_policy = transmitter.start_policy

for i in range(LENGTH):
    #Adversary prediction
    output,ad_pol = adversary.predict(Ad_input)
    #Transmitter prediction
    new_transmission_policy, _ = transmitter.get_policy()
    trans_pol = new_transmission_policy if new_transmission_policy >= 0 \
        else last_transmitter_policy
    #Adversary loss/RL
    trans_bw,trans_vec,past_trans_vecs,ad_pol,ad_choices,ad_bw = Ad_learn(trans_pol,POLICIES,time,NUM_POLICIES,input_states,past_trans_vecs,LOOKBACK,output,optimizer,ad_choices,ad_pol,Ad_input)
    #Tranmsitter loss/RL/Q val updates
    # -- Trans_learn() --
    #accuracy + chart progress
    ad_correct_pol,ad_correct_bw,ad_acc_pol,ad_acc_bw = Ad_accuracy(trans_pol,ad_pol,ad_correct_pol,trans_bw,ad_bw,ad_correct_bw,time)
    trans_correct,trans_acc = Trans_accuracy(trans_bw,ad_bw,trans_correct,time)
    '''
    game_progress()
    '''
    #set up for next timestep
    if i != LENGTH-1:
        time,input_states,next_input,past_inputs,Ad_input = Ad_continue(time,NUM_POLICIES,input_states,trans_pol,next_input,past_inputs,LOOKBACK)
        # -- Trans_continue() --

run_num = len(all_pol_params)
graph_choices(trans_pols,ad_pols,trans_bws,ad_bws,LENGTH,run_num)

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
all_runs[5] = all_notes
save(all_runs,all_ad_acc_pol,all_ad_acc_bw,all_trans_acc,all_acc,trans_pols,trans_bws,all_trans_choices,ad_pols,ad_bws,all_ad_choices)