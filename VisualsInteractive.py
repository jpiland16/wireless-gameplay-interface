import pickle
import matplotlib.pyplot as plt

from statistics import mean, stdev
import scipy.stats as st

from Util import confirm, get_integer, select_option
from Testing import *
from Visuals import Analysis, analyze_game

plt.style.use('seaborn')
plt.rcParams["figure.figsize"] = (8, 6)

class Column:
    def __init__(self, shortname: str, attribute_tree: 'list[str]', 
            check_first_game: bool = False):
        self.shortname = shortname
        self.attribute_tree = attribute_tree
        self.check_first_game = check_first_game

    def extract_from(self, sim_result: 'SimResult | Analysis'):

        if self.check_first_game:
            base = sim_result.games[0]
        else:
            base = sim_result

        tree_position = 0

        while tree_position < len(self.attribute_tree):
            base = getattr(base, self.attribute_tree[tree_position])
            tree_position += 1

        return base

    def __str__(self):
        return self.shortname

# Categorical
FILTER_COLUMNS = [
    Column("Policy Maker", ["policy_maker", "__class__", "__name__"], True),
    Column("Transmitter", ["transmitter", "__class__", "__name__"], True),
    Column("Receiver", ["receiver", "__class__", "__name__"], True),
    Column("Adversary", ["adversary", "__class__", "__name__"], True),
]

# Quantitative, but could be used as categorical
PARAM_COLUMNS = [
    Column("M", ["params", "M"]),
    Column("N", ["params", "N"]),
    Column("T", ["params", "T"]),
    Column("R1", ["params", "R1"]),
    Column("R2", ["params", "R2"]),
    Column("R3", ["params", "R3"]),
    Column("similarity", ["similarity"])
]

# Quantitative
OUT_COLUMNS = [
    Column("transmitter reward", ["score_a"]),
    Column("adversary reward", ["score_b"]),
    Column("adversary accuracy", ["accuracy_b"]),
    Column("switch count", ["switch_count"])
]

def load_file(filename) -> 'list[SimResult]':

    try:
        print("Loading file...")
        results = pickle.load(open(filename, "rb"))
    except:
        results = []
        print("Your file could not be loaded. No results were added.")
    
    return results

def filter_results(results: 'list[SimResult]', filter_choice: Column,
        filter_value: str) -> 'list[SimResult]':
    """
    Return only some of the results from the data.
    """

    filtered_results = []

    for result in results:
        if filter_choice.extract_from(result) == filter_value:
            filtered_results.append(result)

    return filtered_results    

def choose_filter(choices: 'list[Column]') -> 'tuple[Column, str]':
    """
    Allows the user to choose a parameter and desired value to limit the
    amount of data graphed.
    """
    
    print("\nWhich parameter should be used to filter the data?\n")
    filter_choice = select_option(choices)

    filter_value = input("Enter the desired value of this parameter > ")

    return filter_choice, filter_value

def choose_lines(choices: 'list[Column]') -> Column:
    """
    Allow the user to choose which column separates the data into one or more
    lines on the chart.
    """    
    print("\nWhich parameter should be used to separate the graph \n" + 
        "into multiple lines on the chart?\n")
    line_choice = select_option(choices)

    return line_choice

def choose_x_axis(choices: 'list[Column]') -> Column:
    """
    Choose which quantitative input variable will appear on the X-axis.
    """

    print("\nWhich parameter should appear on the X-axis?\n")
    x_choice = select_option(choices)

    return x_choice

def choose_y_axis(choices: 'list[Column]') -> Column:
    """
    Choose which quantitative output variable will appear in the Y-axis.
    """

    print("\nWhich variable should appear on the Y-axis?\n")
    y_choice = select_option(choices)

    return y_choice

def group_by_line(results: 'list[SimResult]', 
        query: Column) -> 'dict[str, list[SimResult]]':

    groups = { }

    for result in results:
        value = str(query.extract_from(result))
        if value not in groups:
            groups[value] = [ ]
        groups[value].append(result)

    return groups

def count(results: 'list[SimResult]'):
    game_count = sum([len(result.games) for result in results])
    return len(results), game_count

def show_count(results: 'list[SimResult]'):
    result_count, game_count = count(results)
    print(f"Current status: {result_count} results, {game_count} games")

def save_graph(data: 'dict[str, dict[str, list]]', use_error_bars: bool,
        x_label: str, y_label: str, title: str, filename: str):

    fmt = "s-"
    markersize = 5

    for label in data:
        # Create a single line
        if use_error_bars:
            (_, caps, _) = plt.errorbar(data[label]["x"], data[label]["y"], 
                data[label]["y_err"], label=label, capsize=3, fmt=fmt,
                    markersize=markersize)

            # NOTE: See https://stackoverflow.com/questions/35915431/top-and-bottom-line-on-errorbar-with-python-and-seaborn
            for cap in caps:
                cap.set_markeredgewidth(1)

        else:
            plt.plot(data[label]["x"], data[label]["y"], label=label, fmt=fmt,
                markersize=markersize)

    plt.title(title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename, dpi=500)
    plt.clf()

def get_data(results: 'list[SimResult]', filter: 'tuple[Column, str]', 
        line_choice: Column, x_choice: Column, y_choice: Column, 
        use_error_bars: bool, confidence_percent: int):

    z_score = st.norm.ppf(0.5 + confidence_percent / 200)

    if filter != None:
        filter_column, value = filter
        filtered_results = filter_results(results, filter_column, value)
    else:
        filtered_results = results
    
    groups = group_by_line(filtered_results, line_choice)

    data = { }

    for label in groups:
        # Create a single line

        group = groups[label]

        x = []
        y = []
        y_err = []

        for result in group:

            x.append(x_choice.extract_from(result))

            all_analyses = []

            for game in result.games:
                all_analyses.append(analyze_game(game))

            points = [y_choice.extract_from(analysis) 
                for analysis in all_analyses]

            y.append(mean(points))

            if use_error_bars:
                y_err.append(stdev(points) * z_score)

        data[label] = {
            "x": x,
            "y": y,
        }

        if use_error_bars:
            data[label]["y_err"] = y_err

    return data  

def get_all_user_choices():

    if confirm("Do you want to filter the data first?"):
        filter_tuple = choose_filter(FILTER_COLUMNS + PARAM_COLUMNS)
    else:
        filter_tuple = None

    line_choice = choose_lines(FILTER_COLUMNS + PARAM_COLUMNS)
    x_choice = choose_x_axis(PARAM_COLUMNS)
    y_choice = choose_y_axis(OUT_COLUMNS)
    use_error_bars = confirm("\nDo you want to use error bars?")

    if use_error_bars:
        confidence_percent = get_integer(
            "Enter the confidence percent (ex: 95)", 0, 99)

    return filter_tuple, line_choice, x_choice, y_choice, use_error_bars, \
            confidence_percent

def get_results():
    
    results = []
    
    while True:
        # Add as many files as you like

        input_filename = input("Enter name of pickle file (leave empty to " + 
            "stop adding data) > ")

        if input_filename == "":
            break

        results += load_file(input_filename)
        show_count(results)

    return results

def main():

    print("Welcome to the interactive graph creator.\n"
        "Please add your results using the prompt below.\n")

    results = get_results()

    while True: 
        # Generate as many graphs as you like

        print()

        filter_tuple, line_choice, x_choice, y_choice, use_error_bars, \
            confidence_percent = get_all_user_choices()

        title = input("\nEnter the title for the graph > ")
        filename = input("Enter the file name to save graph > ")

        data = get_data(
            results = results,
            filter = filter_tuple, # because we already performed the filter
            line_choice = line_choice,
            x_choice = x_choice,
            y_choice = y_choice,
            use_error_bars = use_error_bars,
            confidence_percent = confidence_percent
        )

        save_graph(data, use_error_bars, x_choice.shortname, 
            y_choice.shortname, title, filename)

        if not confirm("Do you want to make another graph with these results?"):
            break

def iget_data(results = None):
    """
    Interactively GET the data (returned as a dict)
    """

    if results == None:
        print("The 'iget_data' function requires data to have already been \n" + 
            "loaded from a pickle file. Try 'results = get_results()' first, \n"
            "then run 'iget_data(results)'.")
        return

    filter_tuple, line_choice, x_choice, y_choice, use_error_bars, \
            confidence_percent = get_all_user_choices()

    return get_data(
            results = results,
            filter = filter_tuple, # because we already performed the filter
            line_choice = line_choice,
            x_choice = x_choice,
            y_choice = y_choice,
            use_error_bars = use_error_bars,
            confidence_percent = confidence_percent
        )

def diy_fast():

    x_choice = PARAM_COLUMNS[6] # Similarity
    y_choice = OUT_COLUMNS[2] # Adversary accuracy

    data = get_data(
        results = pickle.load(open("similarity-test.pkl", "rb")),
        filter = (FILTER_COLUMNS[1], "IntelligentTransmitter"), 
        line_choice = FILTER_COLUMNS[3], # Adversary (different agents)
        x_choice = x_choice,
        y_choice = y_choice,
        use_error_bars = True,
        confidence_percent = 50
    )

    save_graph(
        data, True, x_choice.shortname, y_choice.shortname, 
        title = "IntelligentTransmitter vs Others, 50% confidence",
        filename = "img/test16.png"
    )

if __name__ == "__main__":
    try:
        main()
        # diy_fast()
    except KeyboardInterrupt:
        print()
else:
    print("Welcome to the interactive graph creator! This script is \n" +
        "currently running as part of an interactive shell. \n" + 
        "To get started, try the following commands: \n\n"+
        " - results = get_results() \n"
        " - iget_data(results) \n")
