def invalid_input(check,statement,valid_inputs):
    user_input = check
    if isinstance(valid_inputs,list): 
        exist = valid_inputs.count(check)
        while exist==0:
            print("INVALID INPUT")
            print(statement)
            check = input()
            user_input = check
            exist = valid_inputs.count(check)
    elif valid_inputs == "note":
        while len(check) > 40:
            print("NOTE TOO LONG")
            print(statement)
            check = input()
            user_input = check
    elif valid_inputs == "float":
        while is_float(check)==False:
            print("INVALID INPUT")
            print(statement)
            check = input()
            user_input = check
    else:
        while is_int(check) == False:
            print("INVALID INPUT")
            print(statement)
            check = input()
            user_input = check
    return(user_input)

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True

def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True