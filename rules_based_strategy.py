from yahtzee import Game
from strategy_utils import score_all_rolls, choose_bonus_category_strategic, break_tie_strategic
from constants import CATEGORIES, DEFAULT_TIE_BREAK_ORDER, ALL_ROLLS
from itertools import combinations_with_replacement, combinations, product
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

class RulesBasedStrategy():
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
        From Glenn (2007):
        1. Yahtzee if Y is unused or Yahtzee Joker is applicable
        2. Large Straight if LS or SS is unused
        3. Small straight if SS is unused or both LS and C are unused
        4. A tripleton if the corresponding upper category is unused
        5. Any tripleton if one of 3K, 4K, FH, or C is unused
        6. A doubleton (high preferred) if the corresponding upper
            category is unused
        7. [2 3 4] or [3 4 5] if SS unused or both LS and C are unused
        8. Any doubleton if 3K or C is unused
        9. Any tripleton (high preferred) if Yahtzee is unused or non-zero
        10. A singleton (low preferred) if the corresponding upper category
            is unused, unless more than four upper categories are unused
        11. Any doubleton (high preferred)
        12. A singleton 4, 5, or 6 (high preferred) if 3K, 4K, or C unused
        13. Nothing
        """
        # Get all potential combinations of dice to keept
        dice_values = game_state['dice_values']

        # Filter to columns indices corresponding to available categories
        available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]
        available_categories_names = np.array(CATEGORIES)[available_categories]

        # Get current scores
        potential_scores = game_state['potential_scores']

        # 1. Yahtzee if Y is unused or Yahtzee Joker is applicable
        if potential_scores[CATEGORIES.index('yahtzee')] > 0:
            return [range(0, 5)]
        
        # 2. Large Straight if LS or SS is unused
        if (potential_scores[CATEGORIES.index('lg_straight')] > 0) and (
            'lg_straight' in available_categories_names or 'sm_straight' in available_categories_names
        ):
            return [range(0, 5)] 
        
        # 3. Small straight if SS is unused or both LS and C are unused
        if (potential_scores[CATEGORIES.index('sm_straight')] > 0) and (
            ('sm_straight' in available_categories_names) and
            ('lg_straight' in available_categories_names and 'chance' in available_categories_names)
        ):
            # Find the small straight
            potential_lower_sm_straight = np.arange(min(dice_values), min(dice_values) + 4)
            potential_upper_sm_straight = np.arange(max(dice_values) - 3, max(dice_values) + 1)
            if (np.isin(potential_lower_sm_straight, sorted(dice_values))).all():
                return self.find_dice_index(potential_lower_sm_straight, dice_values)
            elif (np.isin(potential_upper_sm_straight, sorted(dice_values))).all():
                return self.find_dice_index(potential_upper_sm_straight, dice_values)
        
        # 4. A tripleton if the corresponding upper category is unused
        counts = Counter(dice_values)
        max_occur = max(counts.values())
        value_max_occur = sorted([v[0] for v in counts.most_common() if v[1] == max_occur], reverse=True)

        if (max_occur >= 3) and ((value_max_occur[0] - 1) in available_categories):
            return self.find_dice_index([value_max_occur[0]]*max_occur, dice_values) 
            
        # 5. Any tripleton if one of 3K, 4K, FH, or C is unused
        if (max_occur >= 3) and \
                ('3_of_a_kind' in available_categories_names or '4_of_a_kind' in available_categories_names or 'chance' in available_categories_names):
            return self.find_dice_index([value_max_occur[0]]*max_occur, dice_values) 
        
        # 6. A doubleton (high preferred) if the corresponding upper category is unused
        doubles = sorted([v[0] for v in counts.most_common() if v[1] >= 2], reverse=True)
        # Find doubles with available upper section category
        available_upper_doubles = sorted([v for v in doubles if (v-1) in available_categories], reverse=True)
        if len(available_upper_doubles) > 0:
            return self.find_dice_index([available_upper_doubles[0]]*2, dice_values)
        
        # 7. [2 3 4] or [3 4 5] if SS unused or both LS and C are unused
        if ('sm_straight' in available_categories_names) or \
            (('lg_straight' in available_categories_names) and ('chance' in available_categories_names)):
            if np.isin(np.array([2, 3, 4]), sorted(np.unique(dice_values))).all():
                return self.find_dice_index([2, 3, 4], dice_values)
            if np.isin(np.array([3, 4, 5]), sorted(np.unique(dice_values))).all():
                return self.find_dice_index([3, 4, 5], dice_values)
            
        # 8. Any doubleton if 3K or C is unuse
        if (max_occur >= 2) and ('3_of_a_kind' in available_categories_names or 'chance' in available_categories_names):
            return self.find_dice_index([doubles[0]]*2, dice_values)
        
        # 9. Any tripleton (high preferred) if Yahtzee is unused or non-zero
        triples = sorted([v[0] for v in counts.most_common() if v[1] >= 3], reverse=True)
        if (len(triples) > 0) and ('yahtzee' in available_categories_names or game_state['num_yahtzees'] == 1):
            return self.find_dice_index([triples[0]]*3, dice_values)
        
        # 10. A singleton (low preferred) if the corresponding upper category is unused, unless more than four upper categories are unused 
        available_singles = sorted([v[0] for v in counts.most_common() if (v-1) in available_categories])
        num_upper_used = len(available_categories[available_categories <= 5])

        if num_upper_used <= 4 and len(available_singles) > 0:
            return self.find_dice_index([available_singles[0]]*1, dice_values)

        # 11. Any doubleton (high preferred)
        if len(doubles) > 0:
           return self.find_dice_index([doubles[0]]*2, dice_values)
        
        # 12. A singleton 4, 5, or 6 (high preferred) if 3K, 4K, or C unused
        singles = sorted([v[0] for v in counts.most_common() if v >= 4], reverse=True)
        if len(singles) > 0 and (('3_of_a_kind' in available_categories_names) or \
            ('4_of_a_kind' in available_categories_names) or ('chance' in available_categories_names)):
            return self.find_dice_index([singles[0]]*1, dice_values)
        
        # 13. Nothing
        return []
    
    def find_dice_index(self, best_keep_comb, dice_values):
        dice_values_copy = list(dice_values).copy()
        best_keep = []
        for v in best_keep_comb:
            idx = dice_values_copy.index(v)
            dice_values_copy[idx] = -1
            best_keep.append(idx)

        return best_keep


    def choose_category(self, game_state: dict, prioritize_upper_section=True, tie_break_order_idx=DEFAULT_TIE_BREAK_ORDER):
        # 1. Yahtzee
        # 2. Large Straight
        # 3. Small Straight
        # 4. A tripleton in an upper category if it earns the bonus
        # 5. four 5’s or four 6’s in the upper category
        # 6. Four of a Kind but not in first round
        # 7. Three 5’s or three 6’s in the upper category
        # 8. Full House
        # 9. Three of a Kind if the total is at least 22
        # 10. A tripleton in an upper category
        # 11. Three of a Kind
        # 12. Chance if the total is at least 22
        # 13. Doubletons in an upper category (lower preferred)
        # 14. Highest score

        # Filter to columns indices corresponding to available categories
        available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]
        available_categories_names = np.array(CATEGORIES)[available_categories]
        potential_scores = game_state['potential_scores'].copy()
        upper_points_remaining = game_state['upper_points_remaining']

        # 1a. Apply Yahtzee bonus if user rolls second Yahtzee
        if (game_state['num_yahtzees'] == 1) and (potential_scores[11]==100):
            bonus_category, has_bonus_tie = choose_bonus_category_strategic(game_state, 'yahtzee', 11, available_categories, tie_break_order_idx)
            return 'yahtzee', bonus_category, False, has_bonus_tie

        # 1b. Yahtzee, if available
        if potential_scores[CATEGORIES.index('yahtzee')] > 0 and 'yahtzee' in available_categories_names:
            return 'yahtzee', None, False, False

        # 2. Small straight
        if potential_scores[CATEGORIES.index('sm_straight')] > 0 and 'sm_straight' in available_categories_names:
            return 'sm_straight', None, False, False

        # 3. Large straight
        if potential_scores[CATEGORIES.index('lg_straight')] > 0 and 'lg_straight' in available_categories_names:
            return 'lg_straight', None, False, False

        # 4. A tripleton in an upper category if it earns the bonus
        triple_idxs = [i for i, v in enumerate(potential_scores) if i in available_categories and v/3 >= (i+1)]
        if len(triple_idxs) > 0 and upper_points_remaining > 0:
            if potential_scores[triple_idxs[0]] >= upper_points_remaining:
                return CATEGORIES[triple_idxs[0]], None, False, False

        # 5. Four 5’s or four 6’s in the upper category
        if potential_scores[CATEGORIES.index('5s')] == 20:
            return '5s', None, False, False

        if potential_scores[CATEGORIES.index('6s')] == 24:
            return '6s', None, False, False

        # 6. Four of a kind but not in first round
        if (game_state['num_rounds_remaining'] < 13) and ('4_of_a_kind' in available_categories_names) and \
            (potential_scores[CATEGORIES.index('4_of_a_kind')] > 0):
            return '4_of_a_kind', None, False, False

        # 7. Three 5’s or three 6’s in the upper category
        if potential_scores[CATEGORIES.index('5s')] == 15:
            return '5s', None, False, False

        if potential_scores[CATEGORIES.index('6s')] == 18:
            return '6s', None, False, False

        # 8. Full House
        if ('full_house' in available_categories_names) and (potential_scores[CATEGORIES.index('full_house')] > 0):
            return 'full_house', None, False, False

        # 9. Three of a Kind if the total is at least 22
        if ('3_of_a_kind' in available_categories_names) and (potential_scores[CATEGORIES.index('3_of_a_kind')] >= 22):
            return '3_of_a_kind', None, False, False

        # 10. A tripleton in an upper category
        if len(triple_idxs) > 0:
            best_score_idx = triple_idxs[0]
            return CATEGORIES[best_score_idx], None, False, False

        # 11. Three of a Kind
        if ('3_of_a_kind' in available_categories_names) and (potential_scores[CATEGORIES.index('3_of_a_kind')] > 0):
            return '3_of_a_kind', None, False, False
        
        # 12. Chance if the total is at least 22
        if ('chance' in available_categories_names) and (potential_scores[CATEGORIES.index('chance')] >= 22):
            return 'chance', None, False, False

        # 13. Doubletons in an upper category (lower preferred)
        double_idxs = sorted([i for i, v in enumerate(potential_scores) if i in available_categories and v/2 >= (i+1)])
        if len(double_idxs) > 0:
            return CATEGORIES[double_idxs[0]], None, False, False
            
        # 14. Highest score if any category is non-zero
        best_score = max(potential_scores)
        best_score_idxs = [i for i, v in enumerate(potential_scores) if i in available_categories and v == best_score][0]
        if best_score > 0:
            return self.break_tie(best_score_idxs, tie_break_order_idx, is_zero=False), False, False

        # 15. Tie-break forfeited zero
        return self.break_tie(best_score_idxs, tie_break_order_idx, is_zero=False), False, False

    def break_tie(self, best_score_idxs: list, tie_break_order_idx: list, is_zero: bool = True):
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
    g = RulesBasedStrategy(start_seed=3871)
    print(g.run_strategy())