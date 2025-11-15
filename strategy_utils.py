from yahtzee import Game
from constants import CATEGORIES, DEFAULT_TIE_BREAK_ORDER, ALL_ROLLS
from itertools import combinations_with_replacement, combinations, product
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm


def score_all_rolls(g: Game):
    rows = []
    for r in ALL_ROLLS:
        scores = g.score_dice(np.array(r), 0)
        rows.append(scores)
    
    all_rolls = {tuple(sorted(roll)): i for i, roll in enumerate(ALL_ROLLS)}
    all_rolls_scores = np.vstack(rows)

    return all_rolls, all_rolls_scores