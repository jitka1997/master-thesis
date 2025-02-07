# pylint: disable=missing-docstring
from heapq import heappush, heappop
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

PRINT_C = 25


# Not using, just to check if vectoried version is correct when I am paranoid
# Mask is 0 for kept weights, 1 for pruned weights
def calculate_column_gains(W, M):
    n_rows, n_cols = W.shape
    gains = np.zeros((n_cols, n_cols))

    # For each pair of columns i,j
    for i in range(n_cols):
        for j in range(n_cols):
            if i == j:
                continue

            gain = 0
            # For each row, calculate how score changes if we swap columns i and j
            for row in range(n_rows):
                # Original contribution to score
                original = abs(W[row, i]) * M[row, i] + \
                    abs(W[row, j]) * M[row, j]

                # Score after swapping columns
                swapped = abs(W[row, i]) * M[row, j] + \
                    abs(W[row, j]) * M[row, i]

                # Add difference to total gain
                gain += original - swapped

            gains[i, j] = gain

    return gains


def find_kth_best_permutation(G, k):
    n = len(G)
    # Get the initial best permutation
    best_perm = find_optimal_permutation(G)

    # Keep track of seen permutations and solutions
    heap = []
    seen = {tuple(best_perm)}
    solutions = []

    # Add initial solution
    gain = sum(G[i, best_perm[i]] for i in range(n))
    heappush(heap, (-gain, tuple(best_perm)))

    while heap and len(solutions) < k:
        curr_gain, curr_perm = heappop(heap)
        solutions.append(list(curr_perm))

        # Generate all possible swaps from current permutation
        curr_perm = list(curr_perm)
        for i in range(n):
            for j in range(i + 1, n):
                # Create new permutation by swapping
                new_perm = curr_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]

                if tuple(new_perm) not in seen:
                    # Calculate gain for new permutation
                    new_gain = sum(G[i, new_perm[i]] for i in range(n))
                    heappush(heap, (-new_gain, tuple(new_perm)))
                    seen.add(tuple(new_perm))

    return solutions[k-1] if len(solutions) >= k else None


def block_sparsity_pruning(W, block_size=(16, 1), sparsity=0.5):
    rows, cols = W.shape
    block_rows, block_cols = block_size

    # Calculate number of blocks
    n_blocks_row = rows // block_rows
    n_blocks_col = cols // block_cols

    # Reshape into blocks to compute L1 norms
    blocks = W[:n_blocks_row*block_rows, :n_blocks_col*block_cols]
    blocks = blocks.reshape(n_blocks_row, block_rows, n_blocks_col, block_cols)

    # Compute L1 norm for each block
    block_norms = np.abs(blocks).sum(axis=(1, 3))

    # Determine threshold for pruning
    threshold = np.percentile(block_norms, sparsity * 100)

    # Create block mask (1 for kept blocks, 0 for pruned)
    block_mask = (block_norms > threshold).astype(np.float32)

    # Expand block mask to original size
    mask = block_mask.repeat(block_rows, axis=0).repeat(block_cols, axis=1)

    # Handle any remaining rows/cols due to non-divisible dimensions
    if W.shape[0] > mask.shape[0]:
        mask = np.pad(mask, ((0, W.shape[0] - mask.shape[0]), (0, 0)))
    if W.shape[1] > mask.shape[1]:
        mask = np.pad(mask, ((0, 0), (0, W.shape[1] - mask.shape[1])))

    # Apply mask to weights
    W_pruned = W * mask

    return W_pruned, mask


def find_optimal_permutation(G):
    row_ind, col_ind = linear_sum_assignment(G, True)

    # Convert assignment to permutation
    permutation = col_ind

    return permutation

# Calculate gains like in the TETRIS paper

def calculate_row_gains_numpy(W, M):
    # S[i,j] = sum(|W[:,i]| * M[:,j])
    S = np.abs(W) @ M.T

    # Get diagonal elements
    L = np.diag(S)

    # Calculate gain matrix
    G = L[:, None] + L[None, :] - S - S.T

    # Set diagonal to 0 (no gain for swapping with self)
    np.fill_diagonal(G, 0)

    return G


def calculate_column_gains_numpy(W, M):
    # S[i,j] = sum(|W[:,i]| * M[:,j])
    S = np.abs(W).T @ M

    # Get diagonal elements
    L = np.diag(S)

    # Calculate gain matrix
    G = L[:, None] + L[None, :] - S - S.T

    # Set diagonal to 0 (no gain for swapping with self)
    np.fill_diagonal(G, 0)

    return G


def original_tetris_find_optimal_permutation(W, M):
    W_current = W.copy()
    permutation = np.arange(W_current.shape[1])
    max_item = 10
    ii = 0
    while max_item > 1e-6:
        G = calculate_column_gains_numpy(W_current, M)
        # print("G\n", G[:5, 16:20])
        max_item = np.max(G)
        i, j = np.unravel_index(np.argmax(G), G.shape)
        W_current[:, [i, j]] = W_current[:, [j, i]]
        permutation[[i, j]] = permutation[[j, i]]
        # if(ii % 10 == 0):
        #     print(f"ITERATION {ii} MAX ITEM: {max_item:.10f}")
        # ii += 1

    np.set_printoptions(threshold=np.inf)
    return permutation


def original_tetris_pruning(W, block_size=(16, 1), sparsity=0.5, max_iter=10):
    W_current = W.copy()
    original_pruned = 0
    print(f"{'BLOCK':<{PRINT_C}}{'ORIG TETRIS':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")

    for _ in range(max_iter):
        # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = np.abs(W_current)[mask == 0].sum()
        if original_pruned == 0:
            original_pruned = after_block

        # 2. Invert mask to match paper's format (1 = pruned, 0 = kept)
        inverted_mask = 1 - mask

        # 2. Find optimal permutation
        permutation = original_tetris_find_optimal_permutation(
            W_current, inverted_mask)

        # 3. Apply permutation
        W_current = W_current[:, permutation]
        after_tetris = np.abs(W_current)[mask == 0].sum()
        print(f"{after_block:<{PRINT_C}.10f}{after_tetris:<{PRINT_C}.10f}{after_block-after_tetris:<{PRINT_C}.10f}{(after_block-after_tetris)/after_block*100:<{PRINT_C}.10f}{(original_pruned-after_tetris)/original_pruned*100:<{PRINT_C}.10f}")

    return W_current, mask

def add_noise(W, noise_percentage, distribution='normal'):
   noise_level = noise_percentage / 100
   
   if distribution == 'normal':
       noise = np.random.normal(size=W.shape)
   elif distribution == 'uniform':
       noise = np.random.uniform(low=-1, high=1, size=W.shape)
   else:
       raise ValueError("distribution must be 'normal' or 'uniform'")
   
   scaled_noise = noise * np.abs(W) * noise_level
   noisy_W = W + scaled_noise
   
   return noisy_W


def tetris_pruning(W, block_size=(16, 1), sparsity=0.5, max_iter=10, random_swaps=20):
    W_current = W.copy()
    original_pruned = 0
    print(f"{'BLOCK':<{PRINT_C}}{'TETRIS':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")

    for iteration_num in range(max_iter):
        # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = np.abs(W_current)[mask == 0].sum()
        if original_pruned == 0:
            original_pruned = after_block

        # 2. Invert mask to match paper's format (1 = pruned, 0 = kept)
        inverted_mask = 1 - mask

        # 2.5 Add noise
        progress = iteration_num / (max_iter)
        # inv linear: - progress
        # inv sqrt: / np.sqrt(1 + 10 * progress)
        # cosine: * np.cos(progress * np.pi/2)
        W_noisy = add_noise(W_current, 25 - progress * 25, distribution='normal')

        # 3. Calculate gains using inverted mask
        G = calculate_column_gains_numpy(W_noisy, inverted_mask)

        # 4. Find optimal permutation
        permutation = find_optimal_permutation(G)
        # permutation = find_kth_best_permutation(G, 1)

        # np.set_printoptions(threshold=np.inf)
        # print(permutation)

        # 5. Apply permutation
        W_current = W_current[:, permutation]
        after_tetris = np.abs(W_current)[mask == 0].sum()

        print(f"{after_block:<{PRINT_C}.10f}{after_tetris:<{PRINT_C}.10f}{after_block-after_tetris:<{PRINT_C}.10f}{(after_block-after_tetris)/after_block*100:<{PRINT_C}.10f}{(original_pruned-after_tetris)/original_pruned*100:<{PRINT_C}.10f}")

    previous_swap = after_tetris
    print(f"{'AFTER_SWAP':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF AFTER TETRIS %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")

    # Random swaps to improve solution
    for iteration_num in range(random_swaps):
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)

        previous_permutation = permutation.copy()
        # Random swaps
        for _ in range(100):
            i, j = np.random.choice(len(permutation), size=2, replace=False)
            permutation[[i, j]] = permutation[[j, i]]
            W_current[:, [i, j]] = W_current[:, [j, i]]

        # Optimal permutation
        for _ in range(5):
            _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
            inverted_mask = 1 - mask
            G = calculate_column_gains_numpy(W_current, inverted_mask)
            permutation = find_optimal_permutation(G)
            W_current = W_current[:, permutation]

        after_swap = np.abs(W_current)[mask == 0].sum()

        # After swap, diff, diff %, total diff after tetris %, total diff %
        print(f"{after_swap:<{PRINT_C}.10f}{previous_swap-after_swap:<{PRINT_C}.10f}{(previous_swap-after_swap)/previous_swap*100:<{PRINT_C}.10f}{(after_tetris-after_swap)/after_tetris*100:<{PRINT_C}.10f}{(original_pruned-after_swap)/original_pruned*100:<{PRINT_C}.10f}")

        previous_swap = after_swap

    return W_current, mask

def random_swaps_find_mask(W, block_size=(16, 1), sparsity=0.5, max_iter=10):
    W_current = W.copy()
    permutation = np.arange(W_current.shape[1])
    
    _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
    previous_swap = np.abs(W_current)[mask == 0].sum()
    print(f"{'FIRST MASK':<{PRINT_C}}{previous_swap:<{PRINT_C}.10f}")


    for iteration_num in range(max_iter):
        previous_mask = mask.copy()
        previous_permutation = permutation.copy()

        # Random swaps
        for _ in range(1):
            i, j = np.random.choice(len(permutation), size=2, replace=False)
            permutation[[i, j]] = permutation[[j, i]]
            W_current[:, [i, j]] = W_current[:, [j, i]]

        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_mask = np.abs(W_current)[mask == 0].sum()
        if after_mask < previous_swap:
            print("SWAP IMPROVED", after_mask, previous_swap)
            after_swap = after_mask
        else:
            print("SWAP WORSENED", after_mask, previous_swap)
            after_swap = previous_swap
            mask = previous_mask.copy()
            permutation = previous_permutation.copy()
            W_current = W_current[:, permutation]


if __name__ == "__main__":
    original = torch.load("xy.pt").detach().numpy()
    original = original[:200, :2000]
    # print("ORIGINAL:", original, sep="\n")
    # _, original_mask = block_sparsity_pruning(original, block_size=(1, 16))

    BLOCK_SIZE = (1, 32)
    SPARSITY = 0.5
    MAX_ITER = 5000
    RANDOM_SWAPS = 0

    print(
        f"BLOCK SIZE: {BLOCK_SIZE}, SPARSITY: {SPARSITY}, MAX ITER: {MAX_ITER}, SHAPE: {original.shape}")

    # # Apply original tetris
    # reordered, final_mask = original_tetris_pruning(
    #     original, block_size=BLOCK_SIZE, sparsity=SPARSITY, max_iter=MAX_ITER)

    # Apply OUR algorithm
    reordered, final_mask = tetris_pruning(
        original, block_size=BLOCK_SIZE, sparsity=SPARSITY, max_iter=MAX_ITER, random_swaps=RANDOM_SWAPS)

    # # Apply random swaps
    # random_swaps_find_mask(
    #     original, block_size=BLOCK_SIZE, sparsity=SPARSITY, max_iter=MAX_ITER)
