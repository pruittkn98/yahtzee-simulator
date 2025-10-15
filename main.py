import os
import sys
import random
import math
import time
import FreeSimpleGUI as sg


#-------------------------------------------------------------------------------#
#  Works on Python 3                                                            #
#  Uno card game using a GUI interface provided by PySimpleGUI                  #
#  Based on this excellent text based version:                                  #
#  http://code.activestate.com/recipes/580811-uno-text-based/                   #
#  Contains all of the graphics inside the source file as base64 images         #
#  Cards were obtained from Wikipedia                                           #
#        https://en.wikipedia.org/wiki/Uno_(card_game)                          #
#  Up to 4 players... any number can be computer controlled                     #
#  Still needs some work but close enough for fun                               #
#-------------------------------------------------------------------------------#

yellow_color = '#FFAA00'
blue_color = '#5555FF'
red_color = '#FF5555'
green_color = '#55AA55'

import PySimpleGUI as sg
import copy
import random
import numpy as np

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
        self.num_rolls_remaining = 3
        self.num_dice = num_dice
        # Initialize list of dice
        self.dice = np.empty(self.num_dice, dtype=object)
        for i in range(self.num_dice):
            self.dice[i] = Dice(seed = (i + start_seed), num_sides= num_sides)
        self.current_values = np.zeros(self.num_dice)
        self.scores = {
            '1s': 0,
            '2s': 0,
            '3s': 0,
            '4s': 0,
            '5s': 0,
            '6s': 0,
            '3_of_a_kind': 0,
            '4_of_a_kind': 0,
            'full_house': 0,
            'sm_straight': 0,
            'lg_straight': 0,
            'yahtzee': 0,
            'chance': 0
        }

    def score_dice(self):
        self.current_values = np.array([d.get_value() for d in self.dice])

        # Check upper section (1s-6s)
        for n in range(1, 7):
            self.scores[f'{n}s'] = self.check_upper_section(n)

        # Checks three and four of a kind
        for n in [3, 4]:
            self.scores[f'{n}_of_a_kind'] = self.check_three_and_four_of_a_kind(n)

        # Check full house
        self.scores['full_house'] = self.check_full_house()

        # Check straight
        self.scores['sm_straight'] = self.check_sm_straight()
        self.scores['lg_straight'] = self.check_lg_straight()

        # Check Yahtzee
        self.scores['yahtzee'] = self.check_yahtzee()

        # Check chance
        self.scores['chance'] = self.check_chance()

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
            d.roll()

    def clear_scores(self):
        self.scores = {
            '1s': 0,
            '2s': 0,
            '3s': 0,
            '4s': 0,
            '5s': 0,
            '6s': 0,
            '3_of_a_kind': 0,
            '4_of_a_kind': 0,
            'full_house': 0,
            'sm_straight': 0,
            'lg_straight': 0,
            'yahtzee': 0,
            'chance': 0
        }

class Player():
    pass

class Game():
    def __init__(self, player_1_name: str = 'Katie', player_2_name: str = 'Jenny'):
        self.player_1_name = player_1_name
        self.player_2_name = player_2_name

if __name__ == "__main__":
    # a = Dice(seed=15, num_sides = 6)
    # b = Dice(seed=15, num_sides = 6)

    # print(f'A: {a.roll()}, B: {b.roll()}')

    ds = DiceSet(num_dice = 5, num_sides = 6, start_seed = 15)
    ds.roll_dice()
    ds.score_dice()
    print(ds.current_values)
    print(ds.scores)