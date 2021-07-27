import torch
from torch import nn, Tensor
from interface_state import *

past_trans_vecs = []

class RNNAd(nn.Module):

    def __init__(self,NUM_POLICIES, NUM_LAYERS, HIDDEN_DIM):
        super(RNNAd, self).__init__()
        self.input_size = NUM_POLICIES*2
        self.num_layers = NUM_LAYERS
        self.hidden_dim = HIDDEN_DIM
        self.output_size = NUM_POLICIES
        self.rnn = nn.RNN(self.input_size, self.hidden_dim, self.num_layers, batch_first=True)   
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, input):
        output, _ = self.rnn(input)
        output = output.contiguous().view(-1, self.hidden_dim)   
        output = self.fc(output)
        sig = nn.Sigmoid()
        output = sig(output)
        return output

    def predict(self, input):   
        self.zero_grad()
        output = self(input)
        predict = output[-1]
        ad_pol_index = torch.argmax(predict)
        ad_pol_index = ad_pol_index.item()
        ad_pol = ad_pol_index + 1
        return (output, ad_pol)

def Ad_learn(trans_pol,POLICIES,time,NUM_POLICIES,input_states,past_trans_vecs,LOOKBACK,output,optimizer,ad_choices,ad_pol,next_input):
    trans_pol_index = trans_pol - 1
    transmitter = State(POLICIES[trans_pol_index],time)
    trans_bw = transmitter.bw
    trans_vec = []
    for k in range(NUM_POLICIES):
        if k == trans_pol_index:
            trans_vec += [1.0]
        elif input_states[k].bw == trans_bw:
            trans_vec += [0.25]
        else:
            trans_vec += [0.0]
    past_trans_vecs += [trans_vec]
    compare = past_trans_vecs[-LOOKBACK-1:]
    compare = Tensor([compare])
    
    criterion = nn.MSELoss()
    loss = criterion(output,compare)
    loss.backward()
    optimizer.step()

    ad_choices += [ad_pol]
    ad_pol_index = ad_pol - 1
    ad_bw = next_input[ad_pol_index*2]
    return(trans_bw,trans_vec,past_trans_vecs,ad_pol,ad_choices,ad_bw)

def Ad_continue(time,NUM_POLICIES,input_states,trans_pol,next_input,past_inputs,LOOKBACK):
    time += 1
    for m in range(NUM_POLICIES):
        if input_states[m].policy.policy_num == trans_pol:
            input_states[m].chosen = 1
        else:
            input_states[m].chosen = 0
        next_input[m*2+1] = input_states[m].chosen
    past_inputs += [next_input.copy()]
                    
    for k in range(NUM_POLICIES):
        input_states[k] = (input_states[k]).get_next()
        next_input[k*2] = input_states[k].bw
        next_input[k*2+1] = input_states[k].chosen
    if LOOKBACK > time:
        Ad_input = past_inputs.copy()
    else:
        Ad_input = past_inputs[-LOOKBACK:]
    Ad_input += [next_input.copy()]
    Ad_input = Tensor([Ad_input])
    return(time,input_states,next_input,past_inputs,Ad_input)

def one_hot_encode(SIZE,index):
    ret = [0] * SIZE
    ret[index] = 1
    return ret