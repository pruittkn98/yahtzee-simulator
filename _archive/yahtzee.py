import os
import sys
import random
import math
import time

import copy
import random
import numpy as np

CATEGORIES = ['1s', '2s', '3s', '4s', '5s','6s', '3_of_a_kind', '4_of_a_kind', 'full_house', 'sm_straight', 'lg_straight', 'yahtzee', 'chance']

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

class DiceSet():
    def __init__(self, num_dice: int = 5, num_sides: int = 6, start_seed: int = 15):
        self.num_dice = num_dice
        # Initialize list of dice
        self.dice = np.empty(self.num_dice, dtype=object)
        for i in range(self.num_dice):
            self.dice[i] = Dice(seed = (i + start_seed), num_sides= num_sides)
        self.current_values = np.zeros(self.num_dice)
        self.potential_scores = np.array()

    def score_dice(self):
        self.current_values = np.array([d.get_value() for d in self.dice])
        self.potential_scores = np.array()

        # Check upper section (1s-6s)
        for n in range(1, 7):
            self.potential_scores.append(self.check_upper_section(n))

        # Checks three and four of a kind
        for n in [3, 4]:
            self.potential_scores.append(self.check_three_and_four_of_a_kind(n))

        # Check full house
        self.potential_scores.append(self.check_full_house())

        # Check straight
        self.potential_scores.append(self.check_sm_straight())
        self.potential_scores.append(self.check_lg_straight())

        # Check Yahtzee
        self.potential_scores.append(self.check_yahtzee())

        # Check chance
        self.potential_scores.append(self.check_chance())
        return self.potential_scores

    def check_upper_section(self, num: int):
        '''
        Checks how many dice are of given value and returns sum of resulting array
        '''
        vals = self.current_values[self.current_values==num]

        if len(vals) > 0:
            return int(vals.sum())
        else:
            return 0
        
    def check_three_and_four_of_a_kind(self, num: int):
        max_occur= np.bincount(self.current_values).max()
        if max_occur >= num:
            return int(self.current_values.sum())
        else:
            return 0
        
    def check_full_house(self):
        counts = np.unique(np.bincount(self.current_values))
        if len(counts) != 2:
            return 0
    
        elif (np.array([2, 3]) == counts).all():
            return 25
        
        else:
            return 0
        
    def check_sm_straight(self):
        potential_lower_sm_straight = np.arange(min(self.current_values), min(self.current_values) + 4)
        potential_upper_sm_straight = np.arange(max(self.current_values) - 3, max(self.current_values) + 1)
        if (np.isin(sorted(self.current_values), potential_lower_sm_straight)).all() or \
            (np.isin(sorted(self.current_values), potential_upper_sm_straight)).all():
            return 40
        else:
            return 0
        
    def check_lg_straight(self):
        potential_lg_straight = np.arange(min(self.current_values), min(self.current_values) + 5)
        if (sorted(self.current_values)==potential_lg_straight).all():
            return 50
        else:
            return 0

    def check_yahtzee(self):
        if len(np.bincount(self.current_values)) == 1:
            return 50
        else:
            return 0
        
    def check_chance(self):
        return int(self.current_values.sum())

    def roll_dice(self):
        for d in self.dice:
            # Only roll if user has not opted to keep the dice
            if d.is_kept == False:
                d.roll()

    def print_dice(self):
        for i, d in enumerate(self.dice):
            print(f'Die #{i+1}: {d.value}\n')

    def clear(self):
        """
        When turn is over, set all dice to not keep
        """
        for d in self.dice:
            d.is_kept = False

class Player():
    def __init__(self, name: str = 'Katie'):
        self.num_rolls_remaining = 3
        self.name = name
        self.player_scores = np.zeros(len(CATEGORIES))
        self.available_categories =  np.ones(len(CATEGORIES))

    def keep_dice(self, dice_set: DiceSet, keep_indices: list=[]):
        """
        "Keep" dice, determined by input list of indices
        """
        for i, d in enumerate(dice_set):
            if i in keep_indices:
                d.dice[i].keep(True)
            else:
                d.dice[i].keep(False)

    def update_score(self, score_type: str = 'chance', score_value: int = 0):
        score_index = CATEGORIES.index(score_type)
        self.player_scores[score_index] = score_value
        self.available_categories[score_index] = 0
    

class Game():
    def __init__(self, name: str = 'Katie'):
        self.player = Player(name)
        self.dice_set = DiceSet()
        self.rounds_remaining = 13
        self.end_round = False
        self.round_score = 0

    def start_game(self):
        print('Starting game')

    def take_turn(self):
        """
        1. Roll dice
        2. Ask if they want to end the game
        3. Otherwise, ask which dice to keep and then reroll
        """
        self.player.num_rolls_remaining = self.player.num_rolls_remaining - 1
        self.dice_set.roll_dice()
        self.dice_set.print_dice()
        potential_scores = self.dice_set.score_dice()
    
    def process

        if self.player.num_rolls_remaining != 0:

        else:
            self.end_round = True

        if self.end_round == 0:
            print("Which category would you like to apply the score to?")
            print(potential_scores.keys())
            self.dice_set.clear()
            self.player.update_score(score_type = '1s', score_value = 0)


if __name__ == "__main__":
    # a = Dice(seed=15, num_sides = 6)
    # b = Dice(seed=15, num_sides = 6)

    # print(f'A: {a.roll()}, B: {b.roll()}')

    ds = DiceSet(num_dice = 5, num_sides = 6, start_seed = 15)
    ds.roll_dice()
    ds.score_dice()
    print(ds.current_values)
    print(ds.scores)