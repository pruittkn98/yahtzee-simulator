from scripts.utils.yahtzee import Game
from scripts.utils.strategy_utils import choose_bonus_category_strategic, break_tie_strategic
from scripts.utils.constants import CATEGORIES, DEFAULT_TIE_BREAK_ORDER
from collections import Counter
import numpy as np

class RulesBasedStrategy():
    def __init__(self, start_seed=15, 
                        tie_break_order=DEFAULT_TIE_BREAK_ORDER):
        self.start_seed = start_seed
        self.tie_break_order = tie_break_order
        self.tie_break_order_idx = [CATEGORIES.index(v) for v in tie_break_order]
        self.game = Game(start_seed)

    def reset_game(self, start_seed=15, 
                        tie_break_order=DEFAULT_TIE_BREAK_ORDER):
        self.start_seed = start_seed
        self.tie_break_order = tie_break_order
        self.tie_break_order_idx = [CATEGORIES.index(v) for v in tie_break_order]
        self.game = Game(start_seed)

    def choose_dice_to_keep(self, game_state: dict, prioritize_upper_section=False, prioritize_yahtzee=False):
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
            return list(range(0, 5))
        
        # Optional additional strategies: prioritize upper section or prioritize Yahtzee
        # Defaults to standard behavior if either are fulfilled
        if prioritize_upper_section:
            upper_valid = self.check_upper_section(game_state)
            if upper_valid == 0:
                upper_keep_comb = self.choose_dice_to_keep_prioritize_upper(available_categories, dice_values)
                return self.find_dice_index(upper_keep_comb, dice_values)
            
        if prioritize_yahtzee:
            if game_state['num_yahtzees'] < 2:
                yahtzee_comb = self.choose_dice_to_keep_prioritize_yahtzee(available_categories, dice_values)
                return self.find_dice_index(yahtzee_comb, dice_values)
        
        # 2. Large Straight if LS or SS is unused
        if (potential_scores[CATEGORIES.index('lg_straight')] > 0) and (
            'lg_straight' in available_categories_names or 'sm_straight' in available_categories_names
        ):
            return list(range(0, 5))
        
        # 3. Small straight if SS is unused or both LS and C are unused
        if (potential_scores[CATEGORIES.index('sm_straight')] > 0) and (
            ('sm_straight' in available_categories_names) or
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
                ('3_of_a_kind' in available_categories_names or '4_of_a_kind' in available_categories_names or 'full_house' in available_categories_names or 'chance' in available_categories_names):
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
        available_singles = sorted([v[0] for v in counts.most_common() if (v[0]-1) in available_categories])
        num_upper_unused = len([v for v in available_categories if v <= 5])

        if num_upper_unused <= 4 and len(available_singles) > 0:
            return self.find_dice_index([available_singles[0]]*1, dice_values)

        # 11. Any doubleton (high preferred)
        if len(doubles) > 0:
           return self.find_dice_index([doubles[0]]*2, dice_values)
        
        # 12. A singleton 4, 5, or 6 (high preferred) if 3K, 4K, or C unused
        singles = sorted([v[0] for v in counts.most_common() if v[0] >= 4], reverse=True)
        if len(singles) > 0 and (('3_of_a_kind' in available_categories_names) or \
            ('4_of_a_kind' in available_categories_names) or ('chance' in available_categories_names)):
            return self.find_dice_index([singles[0]]*1, dice_values)
        
        # 13. Nothing
        return []
    
    def check_upper_section(self, game_state:dict, currently_scoring=False):
        """
        Checks if
        1. Returns 1 if upper section bonus has been earned
        2. Returns 0 if upper section bonus has not been earned but is attainable
        3. Returns -1 if upper section bonus has not been earned and is not attainable, or if maximum upper score is < 22
        """
        upper_points_remaining = game_state['upper_points_remaining']
        potential_scores = game_state['potential_scores']
        available_upper_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1 and i <= 5]
        current_upper_score = np.sum(game_state['scores'][:6])

        # Additional rules if currently scoring cateogry
        # 1) Return -1 if only 1 upper category is still available and scoring in that category would not surpass threshold for bonus
        # 2) Return -1 if all potential upper category scores are zero
        if currently_scoring:
            max_next_score = 0 if len(available_upper_categories) == 0 else max([v for i, v in enumerate(potential_scores) if i in available_upper_categories])
            if (((current_upper_score + max_next_score) < 63) and len(available_upper_categories) == 1) or \
                max_next_score == 0:
                return -1
            
        # Assume 5 of a kind always applied to Yahtzee
        potential_add_upper_score = sum([(i+1) * 4 for i in available_upper_categories])

        if upper_points_remaining == 0:
            return 1
        
        elif (current_upper_score + potential_add_upper_score) < 63:
            return -1
        
        else:
            return 0
    
    def choose_dice_to_keep_prioritize_upper(self, available_categories: list, dice_values: list):
        """
        Prioritizes upper section bonus. If earned, default to standard behavior.
        """
        # Filter to dice values that match remaining categories
        valid_dice = [d for d in dice_values if (d-1) in available_categories]

        # Reroll all
        if len(valid_dice) == 0:
            return []
        
        # Choose highest expected score (from next immediate roll) among remaining dice (break tie with lower value)
        c = Counter(valid_dice)
        best_count = -1
        best_exp_score = -1
        best_i = -1
        for i in sorted(c.keys()):
            # exp score = current score + expected number of dice of the same value on reroll
            exp_score = c[i] * i + (5 - c[i]) * i * 1/6
            if exp_score > best_exp_score:
                best_count = c[i]
                best_exp_score = exp_score
                best_i = i

        return [best_i] * best_count

    def choose_dice_to_keep_prioritize_yahtzee(self, available_categories: list, dice_values: list):
        """
        Prioritizes getting a Yahtzee. If both Yahtzees are earned, default to standard behavior.
        Keep the number with the highest number of dice. If there's a tie, prioritize upper section availability, and then rank lowest to highest
        """
        # Count number of dice by value
        c = Counter(dice_values)

        best_count = max(c.values())
        best_i = [k for k, v in c.items() if v == best_count]

        if len(best_i) == 1:
            return [best_i[0]] * best_count
        
        # Prioritize upper section
        best_i_upper = [i for i in best_i if (i-1) in available_categories]
        if len(best_i_upper) == 0 and len(best_i):
            return [sorted(best_i, reverse=False)[0]] * best_count
        
        if len(best_i_upper) == 1:
            return [best_i_upper[0]] * best_count
        
        return [sorted(best_i_upper, reverse=False)[0]] * best_count
    
    def find_dice_index(self, best_keep_comb, dice_values):
        """
        Finds indices of dice based on current values
        """
        dice_values_copy = list(dice_values).copy()
        best_keep = []
        for v in best_keep_comb:
            idx = dice_values_copy.index(v)
            dice_values_copy[idx] = -1
            best_keep.append(idx)

        return best_keep


    def choose_category(self, game_state: dict, tie_break_order_idx=DEFAULT_TIE_BREAK_ORDER, prioritize_upper_section=False):
        # 1. Yahtzee
        # 2. Large Straight
        # 3. Small Straight
        # 4. A tripleton in an upper category if it earns the bonus
        # 5. four 5’s or four 6’s in the upper category
        # 6. Four of a Kind but not in first round (Kelly & Liese)
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
        
        # Optional additional strategy: prioritize finishing upper section
        # Yahtzee is already default behavior
        if prioritize_upper_section:
            upper_valid = self.check_upper_section(game_state, currently_scoring=True)
            if upper_valid == 0:
                upper_category_idx, upper_has_tie = self.choose_category_upper(game_state, tie_break_order_idx)
                return CATEGORIES[upper_category_idx], None, upper_has_tie, False
        
        # 2. Large straight
        if potential_scores[CATEGORIES.index('lg_straight')] > 0 and 'lg_straight' in available_categories_names:
            return 'lg_straight', None, False, False

        # 3. Small straight
        if potential_scores[CATEGORIES.index('sm_straight')] > 0 and 'sm_straight' in available_categories_names:
            return 'sm_straight', None, False, False

        # 4. A tripleton in an upper category if it earns the bonus
        triple_idxs = [i for i, v in enumerate(potential_scores) if i in available_categories and v/3 == (i+1)]
        if len(triple_idxs) > 0 and upper_points_remaining > 0:
            if potential_scores[triple_idxs[0]] >= upper_points_remaining:
                return CATEGORIES[triple_idxs[0]], None, False, False

        # 5. Four 5’s or four 6’s in the upper category
        if potential_scores[CATEGORIES.index('5s')] == 20 and '5s' in available_categories_names:
            return '5s', None, False, False

        if potential_scores[CATEGORIES.index('6s')] == 24 and '6s' in available_categories_names:
            return '6s', None, False, False

        # 6. Four of a kind but not in first round
        if (game_state['rounds_remaining'] < 13) and ('4_of_a_kind' in available_categories_names) and \
            (potential_scores[CATEGORIES.index('4_of_a_kind')] > 0):
            return '4_of_a_kind', None, False, False

        # 7. Three 5’s or three 6’s in the upper category
        if potential_scores[CATEGORIES.index('5s')] == 15 and '5s' in available_categories_names:
            return '5s', None, False, False

        if potential_scores[CATEGORIES.index('6s')] == 18 and '6s' in available_categories_names:
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
        best_score = max(potential_scores[available_categories])
        best_score_idxs = [i for i, v in enumerate(potential_scores) if i in available_categories and v == best_score]
        if best_score > 0:
            best_category_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx, is_zero=False)
            return CATEGORIES[best_category_idx], None, has_tie, False

        # 15. Tie-break forfeited zero
        best_category_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx, is_zero=True)
        return CATEGORIES[best_category_idx], None, has_tie, False
    
    def choose_category_upper(self, game_state: dict, tie_break_order_idx=DEFAULT_TIE_BREAK_ORDER):
        """
        Chooses category, prioritizing completing largest contribution toward upper section bonus.
        """
        potential_scores = game_state['potential_scores']
        available_upper_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1 and i <= 5]
        best_upper_score = np.max([v for i, v in enumerate(potential_scores[:6]) if i in available_upper_categories])
            
        best_score_idxs = [i for i, v in enumerate(potential_scores[:6]) if v == best_upper_score and i in available_upper_categories]
        
        best_category_idx, has_tie = break_tie_strategic(best_score_idxs, tie_break_order_idx, is_zero=False)
        return best_category_idx, has_tie

    def run_strategy(self, start_seed=15, tie_break_order=DEFAULT_TIE_BREAK_ORDER, prioritize_upper_section=False, prioritize_yahtzee=False, quiet=True):
        self.reset_game(start_seed=start_seed, tie_break_order=tie_break_order)

        # Iterate through rounds
        for i in range(13):
            if not quiet:
                print(f'Turn #{i + 1}')
                print('   ----- Rolling -----')
            
            for roll_num in range(3):
                if not quiet:
                    print(f'   Roll #{roll_num + 1}')
                
                self.game.roll_dice()
                
                if not quiet:
                    print(f'      Dice values: {self.game.get_game_state()['dice_values']}')
                
                if roll_num < 2:
                    game_state = self.game.get_game_state()
                    best_keep = self.choose_dice_to_keep(game_state, prioritize_upper_section, prioritize_yahtzee)
                    # End turn if best option is to keep all five dice
                    if len(best_keep) == 5:
                        break
                    self.game.keep_dice(best_keep)
                    
                    if not quiet:
                        print(f'      Kept values: {[int(v) for i, v in enumerate(game_state['dice_values']) if i in best_keep]}')

            if not quiet:
                print('   ----- Scoring -----')
            
            # Pick a category when turn is over
            game_state = self.game.get_game_state()
            
            if not quiet:
                print(f'   Final dice values: {game_state['dice_values']}')
            
            best_category, best_bonus_category, has_tie, has_bonus_tie = self.choose_category(game_state, prioritize_upper_section=prioritize_upper_section, tie_break_order_idx=self.tie_break_order_idx)
            self.game.update_score(category=best_category, bonus_category=best_bonus_category, has_tie=has_tie, has_bonus_tie=has_bonus_tie)
            
            if not quiet:
                final_state = self.game.get_game_state()
                print(f'   Category selected: {best_category}')
                print(f'   Bonus category selected: {best_bonus_category}')
                print(f'   Current scores = {final_state['scores']}')
                print(f'   Upper points remaining = {int(final_state['upper_points_remaining'])}')
                print(f'   Total score = {int(final_state['final_score'])}')
                print('-------------------------------------------------------')
            
            self.game.clear()
        
        return self.game.get_game_state()

if __name__ == '__main__':
    g = RulesBasedStrategy(start_seed=6)
    print(g.run_strategy(361, prioritize_upper_section=True, prioritize_yahtzee=False, quiet=False))