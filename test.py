import random
from collections import defaultdict
from itertools import combinations_with_replacement

# produce canonical multiset id mapping
ALL_MULTISETS = list(combinations_with_replacement(range(1,7),5))
IDX = {ms:i for i,ms in enumerate(ALL_MULTISETS)}

import random, copy
from typing import List, Tuple

# ----- adapters: implement or wrap your GameState methods -----
def clone_state(state): ...
def roll_k(state, k): ...
def available_categories(state): ...
def score_hand(state, dice, category): ...
def apply_score(state, category, score): ...

# ----- policy interface -----
class Policy:
    def choose_keep(self, state, dice: List[int], rerolls_left: int) -> Tuple[int,...]:
        raise NotImplementedError
    def choose_category(self, state, dice: List[int]) -> int:
        raise NotImplementedError

def evaluate_policy(state_ctor, policy: Policy, n_games: int) -> float:
    total = 0.0
    for _ in range(n_games):
        st = state_ctor()
        for round_idx in range(13):
            dice = roll_k(st, 5)
            rerolls_left = 2
            while True:
                kept = policy.choose_keep(st, dice, rerolls_left)
                num_reroll = 5 - len(kept)
                if num_reroll == 0 or rerolls_left == 0:
                    final_hand = list(kept) + []  # already 5
                    cat = policy.choose_category(st, final_hand)
                    s = score_hand(st, final_hand, cat)
                    apply_score(st, cat, s)
                    break
                else:
                    new = roll_k(st, num_reroll)
                    dice = list(kept) + new
                    rerolls_left -= 1
        total += getattr(st, "total_score", getattr(st, "score", None))
    return total / n_games


def enumerate_keep_options(multiset):
    # return unique keep outcomes (sorted tuple kept dice) for the multiset
    dice = list(multiset)
    opts = set()
    n = 5
    for mask in range(1<<n):
        kept = tuple(sorted(dice[i] for i in range(n) if (mask>>i)&1))
        opts.add(kept)
    return list(opts)

# baseline policy: greedy final scoring and greedy reroll (simple)
class BaselinePolicy(Policy):
    def choose_keep(self, state, dice, rerolls_left):
        # simple heuristic: keep mode
        from collections import Counter
        c = Counter(dice)
        mode = max(c, key=lambda v:(c[v], v))
        if c[mode] >= 2: return tuple(d for d in dice if d==mode)
        return tuple()  # reroll everything
    def choose_category(self, state, dice):
        cats = available_categories(state)
        best = max(cats, key=lambda c: score_hand(state, dice, c))
        return best

# policy-coded as table for first-roll only:
def random_firstroll_policy():
    table = {}
    for ms in ALL_MULTISETS:
        opts = enumerate_keep_options(ms)
        table[ms] = random.choice(opts)
    return table

def make_policy_from_table(table, baseline):
    class P(Policy):
        def choose_keep(self, state, dice, rerolls_left):
            if rerolls_left==2:
                key = tuple(sorted(dice))
                return table.get(key, baseline.choose_keep(state, dice, rerolls_left))
            return baseline.choose_keep(state, dice, rerolls_left)
        def choose_category(self, state, dice):
            return baseline.choose_category(state, dice)
    return P()

# evolutionary loop
def evolve_firstroll(state_ctor, generations=50, pop=50, eval_games=200, keep_top=5):
    baseline = BaselinePolicy()
    population = [random_firstroll_policy() for _ in range(pop)]
    scores = {}
    for gen in range(generations):
        scored_pop = []
        for i,pol_table in enumerate(population):
            if id(pol_table) not in scores:
                policy = make_policy_from_table(pol_table, baseline)
                scores[id(pol_table)] = evaluate_policy(state_ctor, policy, eval_games)
            scored_pop.append((scores[id(pol_table)], pol_table))
        scored_pop.sort(reverse=True, key=lambda x:x[0])
        print(f"Gen {gen}: best={scored_pop[0][0]:.3f} mean={sum(s for s,_ in scored_pop)/len(scored_pop):.3f}")
        # selection
        top_tables = [t for _,t in scored_pop[:keep_top]]
        # produce new pop by mutating copies of elites
        newpop = []
        while len(newpop) < pop:
            parent = random.choice(top_tables)
            child = {k:v for k,v in parent.items()}
            # mutate: change K entries
            for _ in range(5):
                ms = random.choice(ALL_MULTISETS)
                opts = enumerate_keep_options(ms)
                child[ms] = random.choice(opts)
            newpop.append(child)
        population = newpop
    return make_policy_from_table(scored_pop[0][1], baseline)
