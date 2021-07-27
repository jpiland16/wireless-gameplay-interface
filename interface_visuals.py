import matplotlib.pyplot as plt
import os
import pickle

def add_spacing(max_char,strin,centered=False,html=False):
    extra = max_char - len(strin) 
    if html:
        if centered:
            even = extra//2
            ret = ""
            for i in range(even):
                ret += "&nbsp;"
            ret += strin
            for i in range(even):
                ret += "&nbsp;"
            if extra % 2 == 1:
                ret += "&nbsp"
            return ret
        else:
            for i in range(extra):
                strin += "&nbsp"
            return strin
    else:
        if centered:
            even = extra//2
            ret = ""
            for i in range(even):
                ret += " "
            ret += strin
            for i in range(even):
                ret += " "
            if extra % 2 == 1:
                ret += " "
            return ret
        else:
            for i in range(extra):
                strin += " "
            return strin

#show a certain number of future bandwidths for all policies
#policies = POLICIES_LIST
def options_table(time,policies,future=5):
    time_heading = "   time   |"
    for i in range(time,time+future):
        next_time = str(i) 
        time_heading += add_spacing(5,next_time,True)
        time_heading += "|"
    print(time_heading)
    print("-----------------------------------------------------------------------")
    for i in range(len(policies)):
        next_row = "Policy "
        next_row += str(i+1) 
        next_row = add_spacing(10,next_row)
        next_row += "|"
        for k in range(time,time+future):
            bw = str(policies[i][k])
            next_row += add_spacing(5,bw,True)
            next_row += "|"
        print(next_row)

def game_progress(trans_choices,ad_choices,POLICIES_LIST,time,all_ad_acc_pol=0,all_ad_acc_bw=0,all_trans_acc=0):
    trans_pol = trans_choices[time]
    ad_pol = ad_choices[time]
    trans_bw = POLICIES_LIST[trans_pol-1][time]
    ad_bw = POLICIES_LIST[ad_pol-1][time]
    ad_acc_pol = all_ad_acc_pol[time]
    ad_acc_bw = all_ad_acc_bw[time]
    trans_acc = all_trans_acc[time]
    if time == 0:
        progress_row(trans_pol,trans_bw,ad_pol,ad_bw,time,True,ad_acc_pol,ad_acc_bw,trans_acc)
    else:
        progress_row(trans_pol,trans_bw,ad_pol,ad_bw,time,False,ad_acc_pol,ad_acc_bw,trans_acc)

def progress_row(trans_pol,trans_bw,ad_pol,ad_bw,time,heading,ad_acc_pol=0,ad_acc_bw=0,trans_acc=0):
    if heading == True:
        print("time|Trans Pol|Trans BW|Ad Pol|Ad BW|Ad Pol Acc|Ad BW Acc|Trans Acc")
        print("----------------------------------------------------")
    next_row = ""
    time_heading = str(time)
    next_row += add_spacing(4,time_heading,True)
    next_row += "|"
    add = str(trans_pol)
    next_row += add_spacing(9,add,True)
    next_row += "|"
    add = str(trans_bw)
    next_row += add_spacing(8,add,True)
    next_row += "|"
    add = str(ad_pol)
    next_row += add_spacing(6,add,True)
    next_row += "|"
    add = str(ad_bw)
    next_row += add_spacing(5,add,True)
    next_row += "|"
    add = str(ad_acc_pol)
    add += "%"
    next_row += add_spacing(10,add,True)
    next_row += "|"
    add = str(ad_acc_bw)
    add += "%"
    next_row += add_spacing(9,add,True)
    next_row += "|"
    add = str(trans_acc)
    add += "%"
    next_row += add_spacing(9,add,True)
    next_row += "|"
    print(next_row)

def graph_choices(trans_pols,ad_pols,trans_bws,ad_bws,gamelength,run_num):
    y1 = trans_pols
    y2 = ad_pols
    x = list(range(gamelength))
    fig = plt.figure(num = 1, clear = True, figsize=(20,15))
    ax = fig.add_subplot(2,1,1)
    ax.plot(x, y1, "-", color = 'black')
    ax.plot(x, y2, "-", color = 'red')
    ax.set(xlabel = "Time", ylabel = "Policy #", title = "Transmitter vs. Adversary Policy Choices (Run " + str(run_num) + ")")
    y3 = trans_bws
    y4 = ad_bws
    ax = fig.add_subplot(2,1,2)
    ax.plot(x, y3, "-", color = 'black')
    ax.plot(x, y4, "-", color = 'red')
    ax.set(xlabel = "Time", ylabel = "Bandwidth", title = "Transmitter vs. Adversary Bandwidth Choices (Run " + str(run_num) + ")")
    fig.tight_layout()
    fig.savefig("Graphs/Ad Graph Run " + str(run_num))

def print_runs(all_runs):
    script = '''
    <html lang="en-us">
        <head>
            <title>Data Summary for All Runs</title>
        </head>
        <body>
            <div style="font-family: monospace">Hello World!</div>
        </body>
    </html>
        '''

    rep = ''
    rep += 'Run #|Gamelength|# Pols|# BWs|# Layers|# Nodes|Learning Rate|Lookback|Pol Acc|BW Acc|'
    rep += add_spacing(40,"Notes",True,True)
    rep += "|Graph|"
    rep += "<br>"
    rep += '------------------------------------------------------------------------------------------------------------------------------------'
    for i in range(len(all_runs[0])):
        rep += "<br>"
        add_next = str(i+1)
        rep += add_spacing(5,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[0][i][4])
        rep += add_spacing(10,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[0][i][0])
        rep += add_spacing(6,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[0][i][1])
        rep += add_spacing(5,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[1][i][0])
        rep += add_spacing(8,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[1][i][1])
        rep += add_spacing(7,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[1][i][2])
        rep += add_spacing(13,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[1][i][4])
        rep += add_spacing(8,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[2][i][0])
        rep += add_spacing(7,add_next,True,True)
        rep += "|"
        add_next = str(all_runs[2][i][1])
        rep += add_spacing(6,add_next,True,True)
        rep += "|"
        add_next = all_runs[5][i]
        rep += add_spacing(40,add_next,True,True)
        rep += "|"
        graph_name = "\"Graphs/Ad Graph Run " + str(i+1) + ".png\""
        rep += "<a href=" + graph_name + ">graph</a>"

    script = script.replace("Hello World!", rep)
    with open("runs_summary.html", 'w') as file:
        file.write(script)


ppfilename = 'all_interface_runs.pk'
try:
    with open(ppfilename, 'rb') as fi:
        all_runs = pickle.load(fi)
        print_runs(all_runs)
except FileNotFoundError:
    pass
