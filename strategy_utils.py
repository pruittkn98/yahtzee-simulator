from yahtzee import Game
from constants import CATEGORIES, DEFAULT_TIE_BREAK_ORDER, ALL_ROLLS
from itertools import combinations_with_replacement, combinations, product
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm


def score_all_rolls(g: Game):
    rows = []
    for r in ALL_ROLLS:
        scores = g.score_dice(np.array(r), 0)
        rows.append(scores)
    
    all_rolls = {tuple(sorted(roll)): i for i, roll in enumerate(ALL_ROLLS)}
    all_rolls_scores = np.vstack(rows)

    return all_rolls, all_rolls_scores

def break_tie_strategic(best_score_idxs: list, tie_break_order_idx: list, is_zero: bool = True):
    # Breaks ties, based on specified tie break order
    if len(best_score_idxs) == 1:
        return best_score_idxs[0], False
    
    score_idxs = [(i, tie_break_order_idx.index(i)) for i in best_score_idxs]

    # Reverse priority order if score is zero
    if is_zero:
        score_idxs = sorted(score_idxs, key=lambda x: x[1], reverse=True)
    else:
        score_idxs = sorted(score_idxs, key=lambda x: x[1], reverse=False)

    return score_idxs[0][0], True


def choose_bonus_category_strategic(game_state: dict, category: str, best_score_idx: int, available_categories: list, tie_break_order_idx=DEFAULT_TIE_BREAK_ORDER):
    '''
    Choose bonus category if player rolls second Yahtzee
    Force Joker rules: 
        1. Corresponding upper section category
        2. Three/Four of a kind
        3. Full score on full house/straights or chance
        4. Zero points
    '''
    num_yahtzees = game_state['num_yahtzees']
    new_available_categories = [i for i in available_categories if i != best_score_idx]
    new_available_categories_names = np.array(CATEGORIES)[new_available_categories]
    # Return none if category is not Yahtzee, the player has not yet rolled a Yahtzee, and/or all categories except Yahtzee are filled
    if category != 'yahtzee' or num_yahtzees != 1 or len(new_available_categories) == 0:
        return None
    
    elif len(new_available_categories) == 0:
        best_score_idx = new_available_categories[0]

    else:
        # Check if upper section is available and apply score if it is (forced joker rule)
        new_available_upper_categories = [i for i in new_available_categories if i <= 5]
        new_available_other_categories = [i for i in new_available_categories if i > 5]

        best_score = max(game_state['potential_scores'][new_available_categories])
        best_upper_score = -1 if len(new_available_upper_categories) == 0 else max(game_state['potential_scores'][new_available_upper_categories])

        has_tie = False

        # Apply to the upper section if 1) The respective upper section category is empty (e.g. 5s for 55555) or 2) the respective upper section category is filled but there are no other open lower section categories
        if (best_score == best_upper_score and best_score == 0) or (best_upper_score > 0):
            best_score_idxs = [i for i, x in enumerate(game_state['potential_scores']) if x == best_upper_score and i in new_available_upper_categories]
            best_score_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx)
        
        # Apply to four/three of a kind, if open
        elif '3_of_a_kind' in new_available_categories_names or '4_of_a_kind' in new_available_categories_names:
            best_other_score = max(game_state['potential_scores'][new_available_other_categories])
            best_score_idxs = [i for i, x in enumerate(game_state['potential_scores']) if x == best_other_score and i in new_available_categories]
            best_score_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx)

        # "Steal" full house, small stright, or large straight, or apply to chance
        elif 'full_house' in new_available_categories_names \
            or 'sm_straight' in new_available_categories_names \
            or 'lg_straight' in new_available_categories_names \
            or 'chance' in new_available_categories_names:

            # Recalculate scores (due to stealing)
            new_score_dict = {
                'full_house': {'score': 25, 'index': CATEGORIES.index('full_house')},
                'sm_straight': {'score': 30, 'index': CATEGORIES.index('sm_straight')},
                'lg_straight': {'score': 25, 'index': CATEGORIES.index('lg_straight')},
                'chance': {'score': game_state['potential_scores'][-1:], 'index': CATEGORIES.index('chance')}
            }
            new_scores = {v['index']: v['score'] for k, v in new_score_dict.items() if k in new_available_categories_names}
            best_other_score = max(new_scores.keys())
            best_score_idxs = [i for i, v in new_scores.items() if v == best_other_score]
            best_score_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx)

        # Replace any category with zero
        else:
            available_idxs = [i for i, x in enumerate(game_state['potential_scores']) if i in new_available_categories]
            best_score_idx, has_tie = break_tie_strategic(available_idxs, tie_break_order_idx)
            
    return CATEGORIES[best_score_idx], has_tie