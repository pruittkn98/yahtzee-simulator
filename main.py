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
            self.dice[i] = Dice(seed = (i + self.start_seed), num_sides= num_sides)
        self.current_values = np.zeros(self.num_dice)

    def score_dice(self):
        self.current_values = np.array([d.get_value() for d in self.dice])

    def check_upper_section(self, num: int):
        '''
        Checks how many dice are of given value and returns sum of resulting array
        '''
        vals = np.where(self.current_values == num)

        if vals:
            return vals.sum()
        else:
            return
        
    def check_three_and_four_of_a_kind(self, num: int):
        max_occur= np.bincount(self.current_values).max()
        if max_occur >= num:
            return self.current_values.sum()
        else:
            return

    def roll_dice():
        pass 


class Player():
    pass

class Game():
    def __init__(self, player_1_name: str = 'Katie', player_2_name: str = 'Jenny'):
        self.player_1_name = player_1_name
        self.player_2_name = player_2_name

if __name__ == "__main__":
    a = Dice(seed=15, num_sides = 6)
    b = Dice(seed=15, num_sides = 6)

    print(f'A: {a.roll()}, B: {b.roll()}')