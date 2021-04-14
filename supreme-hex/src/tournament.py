from itertools import combinations, cycle, permutations


def match(p1, p2, state, n, verbose=False):
    turns = cycle((p1, p2))
    states = [state for _ in range(n)]
    wins = {1: 0, 2: 0}

    while states:
        if verbose:
            print(f'{len(states)} states remaining in match')

        player = next(turns)
        actions = player.policies(states)

        new_states = []
        for action, state in zip(actions, states):
            new_state = state.next_state(action)
            winner = new_state.winner()

            if new_state.winner() is None:
                new_states.append(new_state)
            else:
                wins[winner] += 1
        states = new_states

    return wins


def tournament(policies, state, n, verbose=False):
    for (p1, p2) in permutations(policies, 2):
        wins = match(p1, p2, state, n, verbose)
        yield ((p1, wins[1]), (p2, wins[2]))


"""
def tournament(policies, state, n):
    for ((n1, a), (n2, b)) in combinations(policies):
        wa, wb = 0, 0
        for _ in range(n):
            if match(a, b, state)

        for (p1, p2) in [(a, b), (b, a)]:
            winner = match(p1, p2, initial_state)
            if winner == 1:

    pass
"""
