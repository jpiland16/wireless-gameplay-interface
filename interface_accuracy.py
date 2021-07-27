def Ad_accuracy(trans_pol,ad_pol,ad_correct_pol,trans_bw,ad_bw,ad_correct_bw,time):
    if trans_pol == ad_pol:
        ad_correct_pol += 1
    if trans_bw == ad_bw:
        ad_correct_bw += 1
    total = time + 1
    ad_acc_pol = (ad_correct_pol/total) * 100
    ad_acc_pol = round(ad_acc_pol,2)
    ad_acc_pol = str(ad_acc_pol)
    ad_acc_bw = (ad_correct_bw/total) * 100
    ad_acc_bw = round(ad_acc_bw,2)
    ad_acc_bw = str(ad_acc_bw)
    return(ad_correct_pol,ad_correct_bw,ad_acc_pol,ad_acc_bw)

def Trans_accuracy(trans_bw,ad_bw,trans_correct,time):
    if trans_bw != ad_bw:
        trans_correct += 1
    total = time + 1
    trans_acc = (trans_correct/total) * 100
    trans_acc = round(trans_acc,2)
    trans_acc = str(trans_acc)
    return(trans_correct,trans_acc)

