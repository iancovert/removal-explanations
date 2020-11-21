import numpy as np
from rexplain import behavior


def default_batch_size(game):
    '''
    Determine batch size.

    TODO maybe consider the number of features, or the type of model extension.
    '''
    if isinstance(game, behavior.DatasetLossGame):
        return 32
    else:
        return 512


def RemoveIndividual(game, batch_size=None):
    '''Calculate feature attributions by removing individual
    players from the grand coalition.'''
    if batch_size is None:
        batch_size = default_batch_size(game)

    # Setup.
    n = game.players
    S = np.ones((n + 1, n), dtype=bool)
    for i in range(n):
        S[i + 1, i] = 0

    # Evaluate.
    output_list = []
    for i in range(int(np.ceil(len(S) / batch_size))):
        output_list.append(game(S[i*batch_size:(i+1)*batch_size]))
    output = np.concatenate(output_list, axis=0)

    return output[0] - output[1:]


def IncludeIndividual(game, batch_size=None):
    '''Calculate feature attributions by including individual
    players into the empty coalition.'''
    if batch_size is None:
        batch_size = default_batch_size(game)

    # Setup.
    n = game.players
    S = np.zeros((n + 1, n), dtype=bool)
    for i in range(n):
        S[i + 1, i] = 1

    # Evaluate.
    output_list = []
    for i in range(int(np.ceil(len(S) / batch_size))):
        output_list.append(game(S[i*batch_size:(i+1)*batch_size]))
    output = np.concatenate(output_list, axis=0)

    return output[1:] - output[0]


def ShapleyValue(game, batch_size=None, thresh=0.05, verbose=False):
    '''Calculate feature attributions using the Shapley value.'''
    N = 0
    mean = 0
    sum_squares = 0
    converged = False

    # Determine batch size.
    if batch_size is None:
        batch_size = default_batch_size(game)
    arange = np.arange(batch_size)

    # Array to store deltas.
    output = game(np.zeros(game.players, dtype=bool))
    deltas_size = [batch_size, game.players] + list(output.shape)
    deltas = np.zeros(deltas_size)

    while not converged:
        # Sample permutations.
        permutations = np.tile(np.arange(game.players), (batch_size, 1))
        for row in permutations:
            np.random.shuffle(row)
        S = np.zeros((batch_size, game.players), dtype=bool)

        # Unroll permutations.
        prev_value = game(S)
        for i in range(game.players):
            S[arange, permutations[:, i]] = 1
            next_value = game(S)
            deltas[arange, permutations[:, i]] = next_value - prev_value
            prev_value = next_value

        # Update averages (Welford's).
        N += batch_size
        diff = deltas - mean
        mean += np.sum(diff, axis=0) / N
        diff2 = deltas - mean
        sum_squares += np.sum(diff * diff2, axis=0)

        # Check for convergence.
        var = sum_squares / (N ** 2)
        var = np.maximum(var, 0)  # Stability correction for low variance
        std = np.sqrt(var)
        ratio = np.max(
            np.max(std, axis=0) / (np.max(mean, axis=0) - np.min(mean, axis=0)))
        converged = ratio < thresh

        if verbose:
            print('Ratio = {:.4f}'.format(ratio))

    return mean


def BanzhafValue(game, batch_size=None, thresh=0.05, verbose=False):
    '''Calculate feature attributions using the Banzhaf value.'''
    N = 0
    mean = 0
    sum_squares = 0
    converged = False

    # Determine batch size.
    if batch_size is None:
        batch_size = default_batch_size(game)

    # Array to store deltas.
    output = game(np.zeros(game.players, dtype=bool))
    deltas_size = [batch_size, game.players] + list(output.shape)
    deltas = np.zeros(deltas_size)

    while not converged:
        # Sample subsets.
        S = np.random.uniform(size=(batch_size, game.players)) > 0.5

        # Calculate player deltas.
        for i in range(game.players):
            original = S[:, i]
            S[:, i] = 0
            excluded = game(S)
            S[:, i] = 1
            included = game(S)
            S[:, i] = original
            deltas[:, i] = included - excluded

        # Update averages (Welford's).
        N += batch_size
        diff = deltas - mean
        mean += np.sum(diff, axis=0) / N
        diff2 = deltas - mean
        sum_squares += np.sum(diff * diff2, axis=0)

        # Check for convergence.
        var = sum_squares / (N ** 2)
        var = np.maximum(var, 0)  # Stability correction for low variance
        std = np.sqrt(var)
        ratio = np.max(
            np.max(std, axis=0) / (np.max(mean, axis=0) - np.min(mean, axis=0)))
        converged = ratio < thresh

        if verbose:
            print('Ratio = {:.4f}'.format(ratio))

    return mean


def MeanWhenIncluded(game, p=0.5, batch_size=None, thresh=0.05, verbose=False):
    '''Calculate feature attributions through the mean value when
    a player is included in a coalition.'''
    N = 0
    mean = 0
    sum_squares = 0
    converged = False

    # Determine batch size.
    if batch_size is None:
        batch_size = default_batch_size(game)

    while not converged:
        # Sample subsets.
        S = np.random.uniform(size=(batch_size, game.players)) > p

        # Get outcomes, update averages (Welford's).
        output = game(S)[:, np.newaxis]
        if output.ndim > S.ndim:
            S = np.expand_dims(S, -1)

        N = N + np.sum(S.astype(int), axis=0)
        diff = output - mean
        mean += np.sum(S.astype(float) * diff, axis=0) / N
        diff2 = output - mean
        sum_squares += np.sum(S.astype(float) * diff * diff2, axis=0)

        # Check for convergence.
        var = sum_squares / (N ** 2)
        var = np.maximum(var, 0)  # Stability correction for low variance
        std = np.sqrt(var)
        ratio = np.max(
            np.max(std, axis=0) / (np.max(mean, axis=0) - np.min(mean, axis=0)))
        converged = ratio < thresh

        if verbose:
            print('Ratio = {:.4f}'.format(ratio))

    return mean
