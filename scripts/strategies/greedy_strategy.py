from scripts.utils.yahtzee import Game
from scripts.utils.strategy_utils import choose_bonus_category_strategic, break_tie_strategic
from scripts.utils.constants import CATEGORIES, DEFAULT_TIE_BREAK_ORDER, ALL_ROLLS
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

        # Precompute all rolls
        self.all_rolls, self.all_rolls_scores = self._score_all_rolls()

        # Precompute reroll frequencies
        self.reroll_freq = self._find_reroll_frequencies()

    def reset_game(self, start_seed=15, 
                        prioritize_upper_section=True,
                        tie_break_order=DEFAULT_TIE_BREAK_ORDER):
        self.start_seed = start_seed
        self.prioritize_upper_section = prioritize_upper_section
        self.tie_break_order = tie_break_order
        self.tie_break_order_idx = [CATEGORIES.index(v) for v in tie_break_order]
        self.game = Game(start_seed)

    def choose_dice_to_keep(self, game_state: dict, prioritize_upper_section=True):
        """
        Find all combinations of dice that could be kept
        Can keep between 0 and 5 (inclusive)
        """
        # Get all potential combinations of dice to keept
        dice_values = game_state['dice_values']
        upper_points_remaining = game_state['upper_points_remaining']

        # Filter to columns indices corresponding to available categories
        available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]

        all_rolls_scores_copy = self.all_rolls_scores.copy()
        
        # Prioritize getting upper section bonus
        if (prioritize_upper_section == True) and (upper_points_remaining > 0):
            all_rolls_scores_copy[:,:6] += (all_rolls_scores_copy[:,:6]>= upper_points_remaining) * 35

        # Get argmax of each row (assume you'll take the highest score available for the combination)
        max_scores = np.amax(all_rolls_scores_copy[:, available_categories], axis=1)
        
        # Find all potential sets of kept dice
        best_expected_score = -1
        best_keep_comb = ()

        # Loop through number of dice kept
        for i in range(0, 6):
            # Get all potential combinations of kept dice
            keep_combs = list(set(combinations(dice_values, i)))
            reroll_freq_num = self.reroll_freq[i].copy()

            for k in keep_combs:
                expected_score = 0.0

                # Loop through each potential reroll
                for r, p in reroll_freq_num.items():
                    # Get full combination of roll
                    full_roll = tuple(sorted(k + r))
                    # Get ID from array
                    idx = self.all_rolls[full_roll]
                    # Get potential score from index
                    expected_score += max_scores[idx] * p

                if expected_score > best_expected_score:
                    best_expected_score = expected_score
                    best_keep_comb = k

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
            bonus_category, has_bonus_tie = choose_bonus_category_strategic(game_state, 'yahtzee', 11, available_categories, tie_break_order_idx)
            return 'yahtzee', bonus_category, False, has_bonus_tie

        # Prioritize getting upper section bonus
        if (prioritize_upper_section == True) and (upper_points_remaining > 0):
            potential_scores[:6] += (potential_scores[:6]>= upper_points_remaining) * 35

        # Get best score among available categories
        best_score = max(potential_scores[available_categories])

        # Get index of best score
        best_score_idxs = [i for i, x in enumerate(potential_scores) if x == best_score and i in available_categories]

        # Break ties in order specified by tie break (reversed if score is zero)
        if best_score > 0:
            best_score_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx, is_zero=False)
        else:
            best_score_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx, is_zero=True)

        # Return category name
        return CATEGORIES[best_score_idx], None, has_tie, False
    
    def _score_all_rolls(self):
        """
        Precomputes scores for all potential rolls
        """
        rows = []
        for r in ALL_ROLLS:
            scores = self.game.score_dice(np.array(r), 0)
            rows.append(scores)
        
        all_rolls = {tuple(sorted(roll)): i for i, roll in enumerate(ALL_ROLLS)}
        all_rolls_scores = np.vstack(rows)

        return all_rolls, all_rolls_scores
    
    def _find_reroll_frequencies(self):
        """
        Precompute frequency of each reroll combination
        """
        rerolls_freq = {}
        # Loop through number of dice kept
        for i in range(0, 6):
            # Get potential permutations with replacement of rerolled dice
            reroll_combs = list(product(range(1,7),repeat=(5-i)))
            # Count how often each combination appears to get probability
            reroll_counts = Counter(tuple(sorted(t)) for t in reroll_combs)
            rerolls = {r: c/len(reroll_combs) for r, c in reroll_counts.items()}
            # Key is number kept
            rerolls_freq[i] = rerolls

        return rerolls_freq

    def run_strategy(self, start_seed=15, prioritize_upper_section=True, tie_break_order=DEFAULT_TIE_BREAK_ORDER):
        # Reset game
        self.reset_game(start_seed=start_seed, prioritize_upper_section=prioritize_upper_section, tie_break_order=tie_break_order)

        # Iterate through rounds
        while self.game.is_game_over() == False:
            while self.game.is_turn_over() == False:
                self.game.roll_dice()
                if self.game.is_turn_over() == True:
                    break
                best_keep = self.choose_dice_to_keep(self.game.get_game_state(), self.prioritize_upper_section)
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
    g = GreedyStrategy()
    print(g.run_strategy(start_seed=3571))