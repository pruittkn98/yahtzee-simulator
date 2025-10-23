from yahtzee import Game, CATEGORIES
from itertools import combinations_with_replacement, combinations, product
from collections import Counter
import numpy as np
import pandas as pd

def choose_dice_to_keep(game_state: dict, all_rolls, all_rolls_scores):
    """
    Find all combinations of dice that could be kept
    Can keep between 0 and 5 (inclusive)
    """
    # Get all potential combinations of dice to keept
    dice_values = game_state['dice_values']

    # Filter to columns indices corresponding to available categories
    available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]
    
    # Get argmax of each row (assume you'll take the highest score available for the combination)
    max_scores = np.amax(all_rolls_scores[:, available_categories], axis=1)
    
    # Find all potential sets of kept dice
    expected_values = {}

    # Loop through number of dice kept
    for i in range(0, 6):
        # Get all potential combinations of kept dice
        keep_combs = list(set(combinations(dice_values, i)))

        # Get potential combination of rerolled dice
        reroll_combs = list(product(range(1,7),repeat=(5-i)))
        # Count how often each combination appears to get probability
        reroll_counts = Counter(tuple(sorted(t)) for t in reroll_combs)

        for k in keep_combs:
            rerolls = {tuple(sorted(list((k + r)))): {'prob': c/len(reroll_combs)} for r, c in reroll_counts.items()}
            for r in rerolls.keys():
                # Get ID from array
                idx = all_rolls[r]
                # Get score
                rerolls[r]['score'] = max_scores[idx]
                rerolls[r]['expected_score'] = rerolls[r]['score']*rerolls[r]['prob']
            expected_score = np.array([v['expected_score'] for k, v in rerolls.items()]).sum()

            expected_values[k] = expected_score

    # Find which combination has the highest expected score
    best_keep_comb = max(expected_values, key=expected_values.get)

    # Find which dice correspond to these values
    dice_values_copy = list(dice_values).copy()
    best_keep = []
    for v in best_keep_comb:
        idx = dice_values_copy.index(v)
        dice_values_copy[idx] = -1
        best_keep.append(idx)

    return best_keep



def choose_category(game_state: dict):
    # Filter to columns indices corresponding to available categories
    available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]

    # Get best score among available categories
    best_score = max(game_state['potential_scores'][available_categories])

    # Get index of best score (prefer upper section if there's a tie)
    best_score_idx = sorted([i for i, x in enumerate(game_state['potential_scores']) if x == best_score and i in available_categories])[0]

    # Return category name
    return CATEGORIES[best_score_idx]

    

def greedy_strategy(start_seed=15):
    g = Game(start_seed)
    g.roll_dice()

    # Generate all potential roll combinations
    ALL_ROLLS = sorted(list(combinations_with_replacement(range(1,7),5)))

    rows = []
    for r in ALL_ROLLS:
        scores = g.score_dice(np.array(r), 0)
        rows.append(scores)
    
    all_rolls = {tuple(sorted(roll)): i for i, roll in enumerate(ALL_ROLLS)}
    all_rolls_scores = np.vstack(rows)

    while g.is_game_over() == False:
        while g.is_turn_over() == False:
            g.roll_dice()
            best_keep = choose_dice_to_keep(g.get_game_state(), all_rolls, all_rolls_scores)
            if len(best_keep) == 0:
                break
            g.keep_dice(best_keep)

        d = g.get_game_state()
        # pick a category when turn is over
        best_category = choose_category(g.get_game_state())
        g.update_score(category=best_category)
        g.clear()

    return g.get_game_state()['scores']

if __name__ == "__main__":
    df = pd.DataFrame(columns=CATEGORIES + ['final_score'])
    for i in range(5000):
        scores = list(greedy_strategy(i)) + [greedy_strategy(i).sum()]
        df.loc[len(df.index)] = scores
    df.head()
