from dominate import document
from dominate.tags import *

from GameElements import Game
from Testing import SimResult

HEADERS = ["Transmitter", "Adversary", "Score A", "Score B", "B-Acc.", 
            "Sw. Count", "simil.", "T", "M", "N", "R1", "R2", "R3", "link"]

def convert_game_to_html(game: Game):

    with document('Game Results') as doc:

        link(rel='stylesheet', href='../../styles/singlegamestyle.css')

        with table().add(tbody()):

            with tr():
                th()
                [ th(i) for i in range(len(game.state.policy_list)) ]
                [ th(s) for s in ["T", "R", "A"] ]                
        
            for t, round in enumerate(game.state.rounds):

                chosen_band = (game.state.policy_list[game.policy_record[t]]
                                                .get_bandwidth(t))

                with tr():

                    [td(t, cls="timecol")]

                    [td(policy.get_bandwidth(t),
                            cls="bandwidth" + (

                                " correctpol" if i == game.policy_record[t]
                                
                                else (

                                    " correctband" if policy.get_bandwidth(t)
                                        == chosen_band
                                    else ""
                                )

                            )
                        ) 
                    for i, policy in enumerate(game.state.policy_list) ]

                    [ td(x, cls="guess" + (
                        " wrong" if x != chosen_band else ""
                    )) for x in 
                        [ round.transmission_band, 
                        round.receiver_guess,
                        round.adversary_guess ] ]
                    
    return doc.render()

class Analysis:
    def __init__(self, score_a, score_b, accuracy_b, switch_count = 0):
        self.score_a = score_a
        self.score_b = score_b
        self.accuracy_b = accuracy_b
        self.switch_count = switch_count

    def add(self, o: 'Analysis'):
        self.score_a += o.score_a
        self.score_b += o.score_b
        self.accuracy_b += o.accuracy_b
        self.switch_count += o.switch_count

    def divide_by(self, n):
        self.score_a /= n
        self.score_b /= n
        self.accuracy_b /= n
        self.switch_count /= n

    def round_to(self, places):
        self.score_a = round(self.score_a, places)
        self.score_b = round(self.score_b, places)
        self.accuracy_b = round(self.accuracy_b, places)
        self.switch_count = round(self.switch_count, places)


def analyze_game(game: Game):

    analysis = Analysis(
        round(game.state.score_a, 2), 
        round(game.state.score_b, 2), 
        round(100 * (game.state.score_b / game.state.params.R3) 
            / game.state.params.T, 2)
    )

    last_policy = game.policy_record[0]

    for policy in game.policy_record[1:]:
        if policy != last_policy:
            analysis.switch_count += 1
            last_policy = policy

    return analysis


def analyze_result(result: SimResult):

    game_count = len(result.games)
    total_analysis = Analysis(0, 0, 0)
    all_analyses = []
    
    for game in result.games:
        game_analysis = analyze_game(game)
        all_analyses.append(game_analysis)
        total_analysis.add(game_analysis)

    total_analysis.divide_by(game_count)
    total_analysis.round_to(2)

    return total_analysis, all_analyses

def save_games_page(games_analyses: 'list[Analysis]', result: SimResult,
        file_prefix: str, folder: str):
    
    games = result.games

    doc = document('Game List')
    doc.add(link(rel='stylesheet', href='../../styles/gamepagestyle.css'))

    doc.add(table(
            tbody(
                tr(
                    [th(h) for h in HEADERS]
                ),

                [
                    tr(
                        [
                            td(d) for d in [
                                game.transmitter.__class__.__name__,
                                game.adversary.__class__.__name__,
                                single_analysis.score_a,
                                single_analysis.score_b,
                                single_analysis.accuracy_b,
                                single_analysis.switch_count,
                                result.similarity,
                                game.state.params.T,
                                game.state.params.M,
                                game.state.params.N,
                                round(game.state.params.R1, 2),
                                round(game.state.params.R2, 2),
                                round(game.state.params.R3, 2)
                            ] 
                        ] + [td("", a("view game", 
                            href=f"{file_prefix}-game-{index + 1}.html"))]
                    )
                        
                    for index, (single_analysis, game) in enumerate(
                        zip(games_analyses, games))
                ] 
            )
        )
    )

    for index, game in enumerate(games):
        g = convert_game_to_html(game)

        with open(folder + f"{file_prefix}-game-{index + 1}.html", "w") as file:
            file.write(g)

    d = doc.render()

    with open(folder + file_prefix + ".html", "w") as file:
        file.write(d)


def generate_site(results: 'list[SimResult]', folder_name: str):

    doc = document('Results Summary')
    doc.add(link(rel='stylesheet', href='../../styles/summarystyle.css'))
    doc.add(h1("Average results for various parameters"))

    _table = doc.add(table())
    _tbody = _table.add(tbody())
    _tr = _tbody.add(tr())
    _tr.add([th(h) for h in HEADERS])

    for index, result in enumerate(results):

        meta_analysis, games_analyses = analyze_result(result)

        _tr = _tbody.add(tr())

        _tr.add(
            [td(d) for d in [
                result.games[0].transmitter.__class__.__name__,
                result.games[0].adversary.__class__.__name__,
                meta_analysis.score_a,
                meta_analysis.score_b,
                meta_analysis.accuracy_b,
                meta_analysis.switch_count,
                result.similarity,
                round(result.params.T, 2),
                round(result.params.M, 2),
                round(result.params.N, 2),
                round(result.params.R1, 2),
                round(result.params.R2, 2),
                round(result.params.R3, 2)
            ]]
        )

        this_prefix = f"result-{index + 1}"

        _tr.add(td(a("view all games", href=f"{this_prefix}.html")))

        save_games_page(games_analyses, result, this_prefix,
            folder_name)

    with open(folder_name + "_summary.html", "w") as file:
        file.write(doc.render())