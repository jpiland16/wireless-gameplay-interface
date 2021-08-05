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
        results = pickle.load(open(filename, "rb"))
    except:
        results = []
        print("Your file could not be loaded. No results were added.")
    
    return results

def filter_results(results: 'list[SimResult]', filter_choice: Column = None,
        filter_value: str = None) -> 'list[SimResult]':
    """
    Return only some of the results from the data.
    """

    if filter_choice == None:
        print("\nWhich parameter should be used to filter the data?\n")
        filter_choice = select_option(filter_columns + param_columns)

    if filter_value == None:
        filter_value = input("Enter the desired value of this parameter > ")

    filtered_results = []

    for result in results:
        if filter_choice.extract_from(result) == filter_value:
            filtered_results.append(result)

    return filtered_results    

def choose_lines() -> Column:
    """
    Allow the user to choose which column separates the data into one or more
    lines on the chart.
    """    
    print("\nWhich parameter should be used to separate the graph \n" + 
        "into multiple lines on the chart?\n")
    line_choice = select_option(filter_columns + param_columns)

    return line_choice

def choose_x_axis() -> Column:
    """
    Choose which quantitative input variable will appear on the X-axis.
    """

    print("\nWhich parameter should appear on the X-axis?\n")
    x_choice = select_option(param_columns)

    return x_choice

def choose_y_axis() -> Column:
    """
    Choose which quantitative output variable will appear in the Y-axis.
    """

    print("\nWhich variable should appear on the Y-axis?\n")
    y_choice = select_option(out_columns)

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

def save_graph(results: 'list[SimResult]', filter: 'tuple[Column, str]', 
        line_choice: Column, x_choice: Column, y_choice: Column, 
        use_error_bars: bool, confidence_percent: int, title: str, 
        filename: str):

    z_score = st.norm.ppf(0.5 + confidence_percent / 200)

    if filter != None:
        filter_column, value = filter
        filtered_results = filter_results(results, filter_column, value)
    else:
        filtered_results = results
    
    groups = group_by_line(filtered_results, line_choice)

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

        if use_error_bars:
            plt.errorbar(x, y, y_err, label=label)
        else:
            plt.plot(x, y, label=label)

    plt.title(title)
    plt.legend()
    plt.xlabel(x_choice.shortname)
    plt.ylabel(y_choice.shortname)
    plt.savefig(filename, dpi=500)

def main():
    global filter_columns, param_columns, out_columns

    print("Welcome to the interactive graph creator.\n"
        "Please add your results using the prompt below.\n")

    results = []
    
    while True:
        # Add as many files as you like

        input_filename = input("Enter name of pickle file (leave empty to " + 
            "stop adding data) > ")

        if input_filename == "":
            break

        results += load_file(input_filename)
        show_count(results)

    while True: 
        # Generate as many graphs as you like

        filter_columns = FILTER_COLUMNS[:]
        param_columns = PARAM_COLUMNS[:]
        out_columns = OUT_COLUMNS[:]

        print()

        if confirm("Do you want to filter the data first?"):
            filtered_results = filter_results(results)
            show_count(filtered_results)
        else:
            filtered_results = results

        line_choice = choose_lines()
        x_choice = choose_x_axis()
        y_choice = choose_y_axis()
        use_error_bars = confirm("\nDo you want to use error bars?")

        if use_error_bars:
            confidence_percent = get_integer(
                "Enter the confidence percent (ex: 95)", 0, 99)

        title = input("\nEnter the title for the graph > ")
        filename = input("Enter the file name to save graph > ")

        save_graph(
            results = filtered_results,
            filter = None, # because we already performed the filter
            line_choice = line_choice,
            x_choice = x_choice,
            y_choice = y_choice,
            use_error_bars = use_error_bars,
            confidence_percent = confidence_percent,
            title = title,
            filename = filename
        )

        plt.clf()

        if not confirm("Do you want to make another graph with these results?"):
            break

def diy_fast():
    save_graph(
        results = pickle.load(open("similarity-test.pkl", "rb")),
        filter = (FILTER_COLUMNS[1], "IntelligentTransmitter"), 
        line_choice = FILTER_COLUMNS[3], # Adversary (different agents)
        x_choice = PARAM_COLUMNS[6], # Similarity
        y_choice = OUT_COLUMNS[2], # Adversary accuracy
        use_error_bars = True,
        confidence_percent = 70,
        title = "IntelligentTransmitter vs Others, 70% confidence",
        filename = "img/test9.png"
    )

if __name__ == "__main__":
    try:
        main()
        # diy_fast()
    except KeyboardInterrupt:
        print()
