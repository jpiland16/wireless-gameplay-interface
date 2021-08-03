from dominate import document
from dominate.tags import *

from GameElements import Game

def convert_game_to_html(game: Game):

    with document('Game Results') as doc:

        link(rel='stylesheet', href='style.css')

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
