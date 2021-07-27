import pickle
from interface_valid_input import *

def save(all_runs,all_ad_acc_pol,all_ad_acc_bw,all_trans_acc,all_acc,trans_pols,trans_bws,all_trans_choices,ad_pols,ad_bws,all_ad_choices):
    all_acc += [[all_ad_acc_pol[-1:][0]+"%",all_ad_acc_bw[-1:][0]+"%",all_trans_acc[-1:][0]+"%"]]
    all_runs[2] = all_acc
    all_trans_choices += [[trans_pols,trans_bws]]
    all_runs[3] = all_trans_choices
    all_ad_choices += [[ad_pols,ad_bws]]
    all_runs[4] = all_ad_choices
    ppfilename = 'interface_runs.pk'
    with open(ppfilename, 'wb') as fi:
        pickle.dump(all_runs,fi)

def reset_saves():
    statement = "Do you want to reset previous saves? (Y/N)"
    valid_inputs = ["Y","N","y","n"]
    print(statement)
    user_input = input()
    user_input = invalid_input(user_input,statement,valid_inputs)
    if user_input == "Y" or user_input == "y":
        all_runs = []
        all_runs += [[],[],[],[],[],[]]
        ppfilename = 'interface_runs.pk'
        with open(ppfilename, 'wb') as fi:
            pickle.dump(all_runs,fi)