from yahtzee import Game
from constants import CATEGORIES, DEFAULT_TIE_BREAK_ORDER
import numpy as np

class RandomStrategy():
    def __init__(self, start_seed=15, rng_seed = 1, tiered_strategy = True):
        self.start_seed = start_seed
        self.game = Game(start_seed)
        self.rng = np.random.default_rng(seed=rng_seed)
        self.tiered_strategy = tiered_strategy

    def choose_dice_to_keep(self):
        """
        Choose number and combination of dice to keep at random
        -- Tiered random strategy: Choose whether to turn game, choose # of dice to keep and then choose which dice to keep
        -- Purely random strategy: Choose which dice to keep completely at random
        """

        # Choose number of dice
        if self.tiered_strategy:
            # 50/50 chance of ending each turn
            end_turn = self.rng.integers(0, 2)
            if end_turn == 1:
                return list(range(0, 5))
            
            # 1/5 chance of each remaining # of dice
            num_keep_dice =  self.rng.integers(0, 4)
            
            # Choose dice combinations
            return self.rng.choice(range(0, 5), num_keep_dice)

        else:
            best_keep_mask = [self.rng.choice([0, 1]) for i in range(0, 5)]
            return [i for i, v in enumerate(best_keep_mask) if v == 1]


    def choose_category(self, game_state: dict):
        # Filter to columns indices corresponding to available categories
        available_categories = [i for i, x in enumerate(game_state['available_categories']) if x == 1]
        potential_scores = game_state['potential_scores'].copy()

        # Apply Yahtzee bonus if user rolls second Yahtzee
        if (game_state['num_yahtzees'] == 1) and (potential_scores[11]==100):
            bonus_category = self.choose_bonus_category(game_state, 'yahtzee', 11, available_categories)
            return 'yahtzee', bonus_category, False, False
        
        # Choose category at random from non-zero scores in available categories
        non_zero_score_idx = [i for i, v in enumerate(game_state['potential_scores']) if i in available_categories and v > 0]
        if len(non_zero_score_idx) > 0:
            best_score_idx = self.rng.choice(non_zero_score_idx)

        else:
            best_score_idx = self.rng.choice(available_categories)

        # Return category name
        return CATEGORIES[best_score_idx], None, False, False

    def break_tie(self, best_score_idxs: list, tie_break_order_idx: list):
        if len(best_score_idxs) == 1:
            return best_score_idxs[0], False
        
        score_idxs = [(i, tie_break_order_idx.index(i)) for i in best_score_idxs]
        score_idxs = sorted(score_idxs, key=lambda x: x[1])
        return score_idxs[0][0], True


    def choose_bonus_category(self, game_state: dict, category: str, best_score_idx: int, available_categories: list):
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

            # Apply to the upper section if 1) The respective upper section caetgory is empty (e.g. 5s for 55555) or 2) the respective upper section category is filled but there are no other open lower section categories
            if (best_score == best_upper_score and best_score == 0) or (best_upper_score > 0):
                best_score_idxs = [i for i, x in enumerate(game_state['potential_scores']) if x == best_upper_score and i in new_available_upper_categories]
                best_score_idx = self.rng.choice(best_score_idxs, 1)[0]
            
            # If not, apply to lower category according to tiebreak order
            else:
                non_zero_score_idx = [i for i, v in enumerate(game_state['potential_scores']) if i in new_available_other_categories and v > 0]
                if len(non_zero_score_idx) > 0:
                    best_score_idx = self.rng.choice(non_zero_score_idx, 1)[0]

                else:
                    best_score_idx = self.rng.choice(new_available_categories, 1)[0]

        return CATEGORIES[best_score_idx]

    def run_strategy(self):

        while self.game.is_game_over() == False:
            while self.game.is_turn_over() == False:
                self.game.roll_dice()
                if self.game.is_turn_over() == True:
                    break
                
                best_keep = self.choose_dice_to_keep()
                # End turn if best option is to keep all five dice
                if len(best_keep) == 5:
                    break
                self.game.keep_dice(best_keep)

            # Pick a category when turn is over
            best_category, best_bonus_category, has_tie, has_bonus_tie = self.choose_category(self.game.get_game_state())
            self.game.update_score(category=best_category, bonus_category=best_bonus_category, has_tie=has_tie, has_bonus_tie=has_bonus_tie)
            self.game.clear()
        
        return self.game.get_game_state()

if __name__ == '__main__':
    r = RandomStrategy(start_seed=316066, rng_seed = 4294651229, tiered_strategy=True)
    print(r.run_strategy())