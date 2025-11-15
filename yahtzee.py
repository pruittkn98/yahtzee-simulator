import os
import sys
import random
import math
import time

import copy
import random
import numpy as np
from constants import CATEGORIES

class Dice():
    def __init__(self, seed=None, num_sides: int = 6):
        self.seed = seed
        self.is_kept = False
        self.num_sides = num_sides
        self.value = 0
        # Use NumPy RNG for easy variation in seed
        self.rng = np.random.default_rng(seed=seed)

    def roll(self) -> int:
        self.value = int(np.ceil(self.rng.random() * self.num_sides))
        return self.value
    
    def keep(self, is_kept: bool):
        self.is_kept = is_kept

    def get_value(self):
        return self.value

class Game():
    def __init__(self, start_seed: int = 15):
        # Set of dice
        self.dice_set = np.empty(5, dtype=object)
        # Initiates dice, each with different seed (avoid repetition)
        for i in range(5):
            self.dice_set[i] = Dice(seed = (i + start_seed), num_sides=6)
        self.dice_current_values = np.zeros(5)
        # Potential score for current set of dice. Set to empty array
        self.potential_scores = np.array([])

        self.num_rounds_remaining = 13
        self.num_rolls_remaining = 3

        # Array with scores applied
        self.player_scores = np.zeros(len(CATEGORIES))
        # Binary flag if category is still available for scoring
        self.available_categories =  np.ones(len(CATEGORIES))

        self.end_round = False
        self.num_yahtzees = 0
        self.upper_points_remaining = 63

        # Keep track of scoring order for analysis
        self.score_order = []
        self.rolls_per_round = []
        self.ties = []
        self.final_score = 0

    def score_dice(self, values, num_yahtzees):
        """
        Based on current dice values, calculate potential scores for each category
        """
        scores = []

        # Check upper section (1s-6s)
        for n in range(1, 7):
            scores.append(self.check_upper_section(n, values))

        # Checks three and four of a kind
        for n in [3, 4]:
            scores.append(self.check_three_and_four_of_a_kind(n, values))

        # Check full house
        scores.append(self.check_full_house(values))

        # Check straight
        scores.append(self.check_sm_straight(values))
        scores.append(self.check_lg_straight(values))

        # Check Yahtzee
        scores.append(self.check_yahtzee(values, num_yahtzees))

        # Check chance
        scores.append(self.check_chance(values))
        return np.array(scores)

    def update_dice_values(self):
        """
        Retrieves dice current face values
        """
        self.dice_current_values = np.array([d.get_value() for d in self.dice_set])

    def check_upper_section(self, num: int, values: np.array):
        """
        Checks how many dice are of given value and returns sum of resulting array
        """
        vals = values[values==num]

        if len(vals) > 0:
            return int(vals.sum())
        else:
            return 0
        
    def check_three_and_four_of_a_kind(self, num: int, values: np.array):
        """
        Checks for three and four-of-a-kind (bottom section) and returns sum of all dice values if match is found
        """
        max_occur= np.bincount(values).max()
        if max_occur >= num:
            return int(values.sum())
        else:
            return 0
        
    def check_full_house(self, values: np.array):
        """
        Checks for full house and returns 25 if found
        """
        counts = np.unique(np.bincount(values))
        counts = counts[counts>0]
        if len(counts) != 2:
            return 0
    
        elif (np.array([2, 3]) == counts).all():
            return 25
        
        else:
            return 0
        
    def check_sm_straight(self, values: np.array):
        """
        Checks for small straight and returns 30 if found
        """
        # Outline two potential small straights
        potential_lower_sm_straight = np.arange(min(values), min(values) + 4)
        potential_upper_sm_straight = np.arange(max(values) - 3, max(values) + 1)
        if (np.isin(potential_lower_sm_straight, sorted(values))).all() or \
            (np.isin(potential_upper_sm_straight, sorted(values))).all():
            return 30
        else:
            return 0
        
    def check_lg_straight(self, values: np.array):
        """
        Checks for large straight and returns 40 if found
        """
        potential_lg_straight = np.arange(min(values), min(values) + 5)
        if (sorted(values)==potential_lg_straight).all():
            return 40
        else:
            return 0

    def check_yahtzee(self, values: np.array, num_yahtzees: int):
        """
        Checks for Yahtzee (five of a kind) and returns 50 if found and 100 if the user already has one Yahtzee
        """
        if (np.bincount(values).max() == 5) & (num_yahtzees == 0):
            return 50
        elif (np.bincount(values).max() == 5) & (num_yahtzees == 1):
            return 100
        else:
            return 0
        
    def check_chance(self, values: np.array):
        """
        Returns sum of dice values for chance
        """
        return int(values.sum())

    def roll_dice(self):
        """
        Rolls all dice that are not being kept
        """
        if self.num_rolls_remaining <= 0:
            raise ValueError("No rolls remaining")
        for d in self.dice_set:
            # Only roll if user has not opted to keep the dice
            if d.is_kept == False:
                d.roll()
        self.num_rolls_remaining -= 1
        self.update_dice_values()
        self.potential_scores = self.score_dice(self.dice_current_values, self.num_yahtzees)

    def print_dice(self):
        for i, d in enumerate(self.dice_set):
            print(f'Die #{i+1}: {d.value}\n')

    def clear(self):
        """
        Reset at start of turn
        - Reset dice so that all five are rolled
        - Augment number of turns and rolls
        """
        for d in self.dice_set:
            d.is_kept = False
        self.num_rolls_remaining = 3
        self.num_rounds_remaining -= 1
        self.end_round = False

    def keep_dice(self, keep_indices: list=[]):
        """
        Keep dice, determined by input list of indices
        """
        for i, d in enumerate(self.dice_set):
            if i in keep_indices:
                d.keep(True)
            else:
                d.keep(False)

    def update_yahtzee_score(self, bonus_category, score_index):
        '''
        Deals with Yahtzee scoring
        1. If category has already had a zero applied or user has already applied two yahtzees, raise error.
        2. If user has not yet applied any Yahtzees, mark Yahtzee category as used and apply score. Update Yahtzee count if roll used was Yahtzee.
        3. If user has 1 Yahtzee already, add 100 to score and update other chosen category's score
        '''
        if ((self.available_categories[score_index]==0) and (self.num_yahtzees == 0)) \
            or (self.num_yahtzees >= 2):
            raise ValueError(f"Category Yahtzee already used.")
        elif (self.num_yahtzees == 0):
            if (self.potential_scores[score_index] == 50):
                self.num_yahtzees += 1
            self.available_categories[score_index] = 0
            self.player_scores[score_index] = self.potential_scores[score_index]
        else:
            self.num_yahtzees += 1
            self.player_scores[score_index] += 100
            self.available_categories[score_index] = 0

            # Update bonus category
            bonus_score_index = CATEGORIES.index(bonus_category)
            if self.available_categories[bonus_score_index] == 0:
                raise ValueError(f"Category '{bonus_category}' already used.")
            self.player_scores[bonus_score_index] = self.potential_scores[bonus_score_index]
            self.available_categories[bonus_score_index] = 0

    def update_score(self, category: str ='chance', bonus_category=None, has_tie: bool = False, has_bonus_tie: bool = False):
        """ 
        Update score once user has selected dice to keep
        """
        score_index = CATEGORIES.index(category)
        if category == 'yahtzee':
            self.update_yahtzee_score(bonus_category, score_index)
        
        elif (self.available_categories[score_index] == 0):
            raise ValueError(f"Category '{category}' already used.")
        
        else:
            self.player_scores[score_index] = self.potential_scores[score_index]
            self.available_categories[score_index] = 0

        # Update order of bonuses for further analysis
        if bonus_category != None:
            self.score_order.append(category + ', ' + bonus_category)
        else:
            self.score_order.append(category)
        self.rolls_per_round.append((3-self.num_rolls_remaining))
        self.ties.append([has_tie, has_bonus_tie])
        self.update_upper_points_remaining()
        self.update_final_score()
        self.end_round = True

    def update_upper_points_remaining(self):
        upper_scores = self.player_scores[:6].sum()
        self.upper_points_remaining = 0 if upper_scores > 63 else (63 - upper_scores)

    def update_final_score(self):
        '''
        Add 35 to final score if upper section score exceeds 63
        '''
        self.final_score = self.player_scores.sum()
        if self.upper_points_remaining <= 0:
            self.final_score += 35

    def print_potential_scores(self):
        for c, s in zip(CATEGORIES, self.potential_scores):
            print(f'{c}: {s}')

    def is_turn_over(self):
        """
        Check if turn is over
        """
        if self.num_rolls_remaining == 0:
            return True
        return False

    def is_game_over(self):
        """
        Checks if game is over
        """
        if self.num_rounds_remaining == 0:
            return True
        return False
    
    def get_game_state(self):
        return {
            'available_categories': self.available_categories,
            'num_yahtzees': self.num_yahtzees,
            'upper_points_remaining': self.upper_points_remaining,
            'scores': self.player_scores,
            'potential_scores': self.potential_scores,
            'rounds_remaining': self.num_rounds_remaining,
            'rolls_remaining': self.num_rolls_remaining,
            'dice_values': self.dice_current_values,
            'score_order': self.score_order,
            'rolls_per_round': self.rolls_per_round,
            'ties': self.ties,
            'final_score': self.final_score
        }


if __name__ == "__main__":

    # In this example, we roll a small straight on first turn. Reroll two more turns until large straight is obtained.
    g = Game()
    g.roll_dice()
    g.print_dice()
    g.score_dice(g.dice_current_values, g.num_yahtzees)
    g.print_potential_scores()
    g.keep_dice([0, 1, 2, 3])

    g.roll_dice()
    g.print_dice()
    g.score_dice(g.dice_current_values, g.num_yahtzees)
    g.print_potential_scores()
    g.keep_dice([0, 1, 2, 3])

    g.roll_dice()
    g.print_dice()
    g.score_dice(g.dice_current_values, g.num_yahtzees)
    g.print_potential_scores()
    g.keep_dice([0, 1, 2, 3])

    # End round, update score, clear dice
    g.update_score(category='lg_straight')
    g.clear()