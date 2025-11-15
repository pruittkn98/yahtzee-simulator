from yahtzee import Game
from strategy_utils import score_all_rolls
from constants import CATEGORIES, DEFAULT_TIE_BREAK_ORDER, ALL_ROLLS
from itertools import combinations_with_replacement, combinations, product
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

class GreedyStrategy():
    def __init__(self, start_seed=15, 
                        prioritize_upper_section=True,
                        tie_break_order=DEFAULT_TIE_BREAK_ORDER):
        self.start_seed = start_seed
        self.prioritize_upper_section = prioritize_upper_section
        self.tie_break_order = tie_break_order
        self.tie_break_order_idx = [CATEGORIES.index(v) for v in tie_break_order]
        self.game = Game(start_seed)

    def choose_dice_to_keep(self, game_state: dict, all_rolls, all_rolls_scores, prioritize_upper_section=True):
        """
        Find all combinations of dice that could be kept
        Can keep between 0 and 5 (inclusive)
        """
        # Get all potential combinations of dice to keept
        dice_values = game_state['dice_values']
        upper_points_remaining = game_state['upper_points_remaining']

        # Filter to columns indices corresponding to available categories
        available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]
        
        # Prioritize getting upper section bonus
        if (prioritize_upper_section == True) and (upper_points_remaining > 0):
            all_rolls_scores[:,:6] += (all_rolls_scores[:,:6]>= upper_points_remaining) * 35

        # Get argmax of each row (assume you'll take the highest score available for the combination)
        max_scores = np.amax(all_rolls_scores[:, available_categories], axis=1)
        
        # Find all potential sets of kept dice
        expected_values = {}

        # Loop through number of dice kept
        for i in range(0, 6):
            # Get all potential combinations of kept dice
            keep_combs = list(set(combinations(dice_values, i)))

            # Get potential permutations with replacement of rerolled dice
            reroll_combs = list(product(range(1,7),repeat=(5-i)))

            # Count how often each combination appears to get probability
            reroll_counts = Counter(tuple(sorted(t)) for t in reroll_combs)

            for k in keep_combs:
                rerolls = {tuple(sorted(list((k + r)))): {'prob': c/len(reroll_combs)} for r, c in reroll_counts.items()}
                for r in rerolls.keys():
                    # Get ID from array
                    idx = all_rolls[r]
                    # Get potential score from index
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

    def choose_category(self, game_state: dict, prioritize_upper_section=True, tie_break_order_idx=DEFAULT_TIE_BREAK_ORDER):
        # Filter to columns indices corresponding to available categories
        available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]
        potential_scores = game_state['potential_scores'].copy()
        upper_points_remaining = game_state['upper_points_remaining']

        # Apply Yahtzee bonus if user rolls second Yahtzee
        if (game_state['num_yahtzees'] == 1) and (potential_scores[11]==100):
            bonus_category, has_bonus_tie = self.choose_bonus_category(game_state, 'yahtzee', 11, available_categories, tie_break_order_idx)
            return 'yahtzee', bonus_category, False, has_bonus_tie

        # Prioritize getting upper section bonus
        if (prioritize_upper_section == True) and (upper_points_remaining > 0):
            potential_scores[:6] += (potential_scores[:6]>= upper_points_remaining) * 35

        # Get best score among available categories
        best_score = max(game_state['potential_scores'][available_categories])

        # Get index of best score
        best_score_idxs = [i for i, x in enumerate(game_state['potential_scores']) if x == best_score and i in available_categories]

        # Break ties in order specified by tie break
        best_score_idx, has_tie = self.break_tie(best_score_idxs, tie_break_order_idx)

        # Return category name
        return CATEGORIES[best_score_idx], None, has_tie, False

    def break_tie(self, best_score_idxs: list, tie_break_order_idx: list):
        if len(best_score_idxs) == 1:
            return best_score_idxs[0], False
        
        score_idxs = [(i, tie_break_order_idx.index(i)) for i in best_score_idxs]
        score_idxs = sorted(score_idxs, key=lambda x: x[1])
        return score_idxs[0][0], True


    def choose_bonus_category(self, game_state: dict, category: str, best_score_idx: int, available_categories: list, tie_break_order_idx=DEFAULT_TIE_BREAK_ORDER):
        '''
        Choose bonus category if player rolls second Yahtzee
        Force Joker rules: Search upper section first. If not filled, apply to best lower section option
        '''
        num_yahtzees = game_state['num_yahtzees']
        new_available_categories = [i for i in available_categories if i != best_score_idx]
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

            # Apply to the upper section if 1) The respective upper section caetgory is empty (e.g. 5s for 55555) or 2) the respective upper section category is filled but there are no other open lower section categories
            if (best_score == best_upper_score and best_score == 0) or (best_upper_score > 0):
                best_score_idxs = [i for i, x in enumerate(game_state['potential_scores']) if x == best_upper_score and i in new_available_upper_categories]
                best_score_idx, has_tie = self.break_tie(best_score_idxs, tie_break_order_idx)
            
            # If not, apply to lower category according to tiebreak order
            else:
                best_other_score = max(game_state['potential_scores'][new_available_other_categories])
                best_score_idxs = [i for i, x in enumerate(game_state['potential_scores']) if x == best_other_score and i in new_available_categories]
                best_score_idx, has_tie = self.break_tie(best_score_idxs, tie_break_order_idx)

        return CATEGORIES[best_score_idx], has_tie

    def run_strategy(self):
        # Get score for all potential rolls
        all_rolls, all_rolls_scores = score_all_rolls(self.game)

        while self.game.is_game_over() == False:
            while self.game.is_turn_over() == False:
                self.game.roll_dice()
                if self.game.is_turn_over() == True:
                    break
                best_keep = self.choose_dice_to_keep(self.game.get_game_state(), all_rolls, all_rolls_scores, self.prioritize_upper_section)
                # End turn if best option is to keep all five dice
                if len(best_keep) == 5:
                    break
                self.game.keep_dice(best_keep)

            # Pick a category when turn is over
            best_category, best_bonus_category, has_tie, has_bonus_tie = self.choose_category(self.game.get_game_state(), self.prioritize_upper_section, self.tie_break_order_idx)
            self.game.update_score(category=best_category, bonus_category=best_bonus_category, has_tie=has_tie, has_bonus_tie=has_bonus_tie)
            self.game.clear()
        
        return self.game.get_game_state()

if __name__ == '__main__':
    g = GreedyStrategy(start_seed=3871)
    print(g.run_strategy())