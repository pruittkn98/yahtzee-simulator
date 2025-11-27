from itertools import combinations_with_replacement

CATEGORIES = ['1s',
               '2s', 
               '3s', 
               '4s', 
               '5s',
               '6s', 
               '3_of_a_kind',
                '4_of_a_kind', 
                'full_house',
                'sm_straight', 
                'lg_straight', 
                'yahtzee', 
                'chance']

DEFAULT_TIE_BREAK_ORDER = [ 'yahtzee', 
                                    'sm_straight', 
                                    'lg_straight', 
                                    '4_of_a_kind', 
                                    '3_of_a_kind',
                                    'full_house',
                                    '1s',
                                    '2s', 
                                    '3s', 
                                    '4s', 
                                    '5s',
                                    '6s', 
                                    'chance']

ALL_ROLLS = sorted(list(combinations_with_replacement(range(1,7),5)))
