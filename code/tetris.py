# pylint: disable=missing-docstring
from heapq import heappush, heappop
import time
import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PRINT_C = 25
TOLERANCE = 1e-4


# Not using, just to check if vectoried version is correct when I am paranoid
# Mask is 0 for kept weights, 1 for pruned weights
def calculate_column_gains_slow(W, M):
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
    best_perm_tensor = find_optimal_permutation(G)
    best_perm = best_perm_tensor.cpu().tolist()

    # Keep track of seen permutations and solutions
    heap = []
    seen = {tuple(best_perm)}
    solutions = []

    # Move G to CPU for the Python loop.
    G_cpu = G.cpu().numpy()

    gain = sum(G_cpu[i, best_perm[i]] for i in range(n))
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
                    new_gain = sum(G_cpu[i, new_perm[i]] for i in range(n))
                    heappush(heap, (-new_gain, tuple(new_perm)))
                    seen.add(tuple(new_perm))

    if len(solutions) >= k:
        # Convert the final list back to a tensor on the original device
        return torch.tensor(solutions[k-1], device=G.device)
    return None


def block_sparsity_pruning(W, block_size=(1, 8), sparsity=0.5):
    rows, cols = W.shape
    block_rows, block_cols = block_size

    n_blocks_row = rows // block_rows
    n_blocks_col = cols // block_cols

    # View instead of reshape where possible, but reshape is safer if non-contiguous
    blocks = W[:n_blocks_row*block_rows, :n_blocks_col*block_cols]
    blocks = blocks.reshape(n_blocks_row, block_rows, n_blocks_col, block_cols)

    # Compute sum for each block. Shape: (n_blocks_row, n_blocks_col)
    block_norms = blocks.abs().sum(dim=(1, 3))

    # Calculate the quantile for EACH row individually
    # torch.quantile takes a value between 0.0 and 1.0
    thresholds = torch.quantile(block_norms, sparsity, dim=1, keepdim=True)

    # Create block mask (compare each block to its row's specific threshold)
    block_mask = (block_norms > thresholds).float()

    # Expand block mask to original size using repeat_interleave
    mask = block_mask.repeat_interleave(block_rows, dim=0).repeat_interleave(block_cols, dim=1)

    # Handle padding
    pad_bottom = rows - mask.shape[0]
    pad_right = cols - mask.shape[1]
    
    if pad_bottom > 0 or pad_right > 0:
        mask = F.pad(mask, (0, pad_right, 0, pad_bottom), value=0.0)

    W_pruned = W * mask

    return W_pruned, mask


# def block_sparsity_pruning(W, block_size=(16, 1), sparsity=0.5):
#     rows, cols = W.shape
#     block_rows, block_cols = block_size

#     # Calculate number of blocks
#     n_blocks_row = rows // block_rows
#     n_blocks_col = cols // block_cols

#     # Reshape into blocks to compute L1 norms
#     blocks = W[:n_blocks_row*block_rows, :n_blocks_col*block_cols]
#     blocks = blocks.reshape(n_blocks_row, block_rows, n_blocks_col, block_cols)

#     # Compute L1 norm for each block
#     block_norms = np.abs(blocks).sum(axis=(1, 3))

#     # Determine threshold for pruning
#     threshold = np.percentile(block_norms, sparsity * 100)

#     # Create block mask (1 for kept blocks, 0 for pruned)
#     block_mask = (block_norms > threshold).astype(np.float32)

#     # Expand block mask to original size
#     mask = block_mask.repeat(block_rows, axis=0).repeat(block_cols, axis=1)

#     # Handle any remaining rows/cols due to non-divisible dimensions
#     if W.shape[0] > mask.shape[0]:
#         mask = np.pad(mask, ((0, W.shape[0] - mask.shape[0]), (0, 0)))
#     if W.shape[1] > mask.shape[1]:
#         mask = np.pad(mask, ((0, 0), (0, W.shape[1] - mask.shape[1])))

#     # Apply mask to weights
#     W_pruned = W * mask

#     return W_pruned, mask

def find_optimal_permutation(G):
    G_cpu = G.cpu().numpy()
    
    # Column indices are the optimal permutation of columns that minimizes the total gain
    row_ind, col_ind = linear_sum_assignment(G_cpu, True)

    # Convert assignment back to a tensor on the original device
    permutation = torch.tensor(col_ind, device=G.device)
    return permutation

# Calculate gains like in the TETRIS paper

def calculate_row_gains(W, M):
    # S[i,j] = sum(|W[:,i]| * M[:,j])
    S = torch.matmul(W.abs(), M.float().T)

    L = torch.diagonal(S)

    # Calculate gain matrix
    G = L.unsqueeze(1) + L.unsqueeze(0) - S - S.T

    # Set diagonal to 0 (no gain for swapping with self)
    G.fill_diagonal_(0)

    return G

def calculate_column_gains(W, M):
    # S[i,j] = sum(|W[:,i]| * M[:,j])
    S = torch.matmul(W.abs().T, M.float())

    # Get diagonal elements
    L = torch.diagonal(S)

    # Calculate gain matrix
    G = L.unsqueeze(1) + L.unsqueeze(0) - S - S.T

    # Set diagonal to 0 (no gain for swapping with self)
    G.fill_diagonal_(0)
    return G


def original_tetris_find_optimal_permutation(W, M):
    W_current = W.clone()
    device = W.device
    permutation = torch.arange(W_current.shape[1], device=device)
    max_item = 10
    max_internal_swaps = 1000 
    swap_count = 0
    
    while max_item > 1e-5 and swap_count < max_internal_swaps:
        G = calculate_column_gains(W_current, M)
        max_item = torch.max(G).item()
        
        # PyTorch argmax returns a flat index, we need to convert it to 2D
        idx = torch.argmax(G).item()
        i = idx // G.shape[1]
        j = idx % G.shape[1]
        
        if i == j:
            break

        # Swap columns
        W_current[:, [i, j]] = W_current[:, [j, i]]
        permutation[[i, j]] = permutation[[j, i]]
        swap_count += 1

    return permutation

def original_tetris_pruning(W, block_size=(16, 1), sparsity=0.5, max_iter=10, verbose=True):
    t0 = time.perf_counter()
    best_time_relative = 0.0
    history = []

    W_current = W.clone()
    _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
    original_pruned = W_current.abs()[mask == 0].sum().item()
    history.append((best_time_relative, original_pruned))
    best_score_so_far = original_pruned
    global_perm = torch.arange(W.shape[1], device=W.device)

    if verbose:
        print(f"{'BLOCK':<{PRINT_C}}{'ORIG TETRIS':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")

    for iteration_num in range(max_iter):
        if (iteration_num + 1) % 10 == 0:
            print(f"Tetris iteration {iteration_num + 1}/{max_iter}")

         # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = W_current.abs()[mask == 0].sum().item()
        if original_pruned == 0:
            original_pruned = after_block

        # 2. Invert mask to match paper's format (1 = pruned, 0 = kept)
        inverted_mask = 1.0 - mask

        # 3. Find optimal permutation
        permutation = original_tetris_find_optimal_permutation(W_current, inverted_mask)
        global_perm = global_perm[permutation]
        
        # 3. Apply permutation
        W_current = W_current[:, permutation]
        after_tetris = W_current.abs()[mask == 0].sum().item()

        if after_tetris < best_score_so_far:
            best_score_so_far = after_tetris
            best_time_relative = time.perf_counter() - t0
            history.append((best_time_relative, best_score_so_far))

        if verbose:
            print(f"{after_block:<{PRINT_C}.10f}{after_tetris:<{PRINT_C}.10f}{after_block-after_tetris:<{PRINT_C}.10f}{(after_block-after_tetris)/after_block*100:<{PRINT_C}.10f}{(original_pruned-after_tetris)/original_pruned*100:<{PRINT_C}.10f}")

    return W_current, mask, global_perm, best_time_relative, history

def add_noise(W, noise_percentage, distribution='normal'):
    noise_level = noise_percentage / 100.0
    
    if distribution == 'normal':
        noise = torch.randn_like(W)
    elif distribution == 'uniform':
        noise = torch.empty_like(W).uniform_(-1.0, 1.0)
    else:
        raise ValueError("distribution must be 'normal' or 'uniform'")
    
    scaled_noise = noise * W.abs() * noise_level
    noisy_W = W + scaled_noise
    
    return noisy_W


def tetris_pruning(W, block_size=(1, 8), sparsity=0.5, max_iter=10, random_swaps=20, verbose=True, noise_scale=1.5):
    t0 = time.perf_counter()
    best_time_relative = 0.0
    history = []

    W_current = W.clone()

    # # Add sorting by column norm at the start
    # col_norms = W_current.abs().sum(dim=0)
    # sorted_permutation = torch.argsort(col_norms, descending=True)
    # W_current = W_current[:, sorted_permutation]
    # global_perm = sorted_permutation.clone()

    _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
    original_pruned = W_current.abs()[mask == 0].sum().item()
    history.append((best_time_relative, original_pruned))
    best_score_so_far = original_pruned
    global_perm = torch.arange(W.shape[1], device=W.device)

    if verbose:
        print(f"{'BLOCK':<{PRINT_C}}{'TETRIS':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")

    for iteration_num in range(max_iter):
        if (iteration_num + 1) % 10 == 0 and verbose:
            print(f"Tetris iteration {iteration_num + 1}/{max_iter}")

        # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = W_current.abs()[mask == 0].sum().item()

        # 2. Invert mask to match paper's format (1 = pruned, 0 = kept)
        inverted_mask = 1.0 - mask

        # 2.5 Add noise
        # progress = iteration_num / (max_iter)
        # inv linear: - progress
        # inv sqrt: / np.sqrt(1 + 10 * progress)
        # cosine: * np.cos(progress * np.pi/2)
        # W_noisy = add_noise(W_current, 25 - progress * 25, distribution='normal')
        # MULTIPLICATIVE noise
        # progress = iteration_num / max_iter
        # noise_scale = 1 * (1.0 - progress)
        # noise_factor = np.random.normal(loc=1.0, scale=noise_scale, size=W_current.shape)
        # noise_factor = np.clip(noise_factor, a_min=0.1, a_max=None)

        # LogNormal Noise natively in PyTorch
        sigma = noise_scale * (1.0 - iteration_num / max_iter)
        if sigma > 0:
            # log_normal_ requires an empty tensor to fill
            noise_factor = torch.empty_like(W_current).log_normal_(mean=0.0, std=sigma)
        else:
            noise_factor = torch.ones_like(W_current)

        W_noisy = W_current * noise_factor

        # 3. Calculate gains using inverted mask
        G = calculate_column_gains(W_noisy, inverted_mask)

        # 4. Find optimal permutation
        permutation = find_optimal_permutation(G)
        global_perm = global_perm[permutation]
        # permutation = find_kth_best_permutation(G, 1)

        # 5. Apply permutation
        W_current = W_current[:, permutation]
        after_tetris = W_current.abs()[mask == 0].sum().item()

        if after_tetris < best_score_so_far:
            best_score_so_far = after_tetris
            best_time_relative = time.perf_counter() - t0
            history.append((best_time_relative, best_score_so_far))

        if verbose:
            print(f"{after_block:<{PRINT_C}.10f}{after_tetris:<{PRINT_C}.10f}{after_block-after_tetris:<{PRINT_C}.10f}{(after_block-after_tetris)/after_block*100:<{PRINT_C}.10f}{(original_pruned-after_tetris)/original_pruned*100:<{PRINT_C}.10f}")

    previous_swap = after_tetris
    previous_global_perm = global_perm.clone()
    history.append((time.perf_counter() - t0, after_tetris))
    after_tetris_index_in_history = len(history) - 1

    if verbose:
        print(f"{'AFTER_SWAP':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF AFTER TETRIS %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")

    # Random swaps to improve solution
    for iteration_num in range(random_swaps):
        if (iteration_num + 1) % 10 == 0 and verbose:
            print(f"Random swap iteration {iteration_num + 1}/{random_swaps}")
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)

        previous_permutation = permutation.clone()
        previous_W = W_current.clone()
        
        # Random swaps
        for _ in range(50):
            # Get a 1D tensor containing 2 random indices
            indices = torch.randperm(len(permutation), device=W.device)[:2]
            
            # Get the reversed version of those indices
            rev_indices = torch.flip(indices, dims=[0])
            
            permutation[indices] = permutation[rev_indices]
            global_perm[indices] = global_perm[rev_indices]
            W_current[:, indices] = W_current[:, rev_indices]

        # Optimal permutation
        for _ in range(5):
            _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
            inverted_mask = 1.0 - mask
            G = calculate_column_gains(W_current, inverted_mask)
            permutation = find_optimal_permutation(G)
            global_perm = global_perm[permutation]
            W_current = W_current[:, permutation]

        after_swap = W_current.abs()[mask == 0].sum().item()
        
        if (previous_swap - after_swap > TOLERANCE):
            if verbose:
                print(f"{after_swap:<{PRINT_C}.10f}{previous_swap-after_swap:<{PRINT_C}.10f}{(previous_swap-after_swap)/previous_swap*100:<{PRINT_C}.10f}{(after_tetris-after_swap)/after_tetris*100:<{PRINT_C}.10f}{(original_pruned-after_swap)/original_pruned*100:<{PRINT_C}.10f}")
            previous_swap = after_swap
            best_time_relative = time.perf_counter() - t0
            history.append((best_time_relative, after_swap))
        else:
            W_current = previous_W.clone()
            permutation = previous_permutation.clone()
            global_perm = previous_global_perm.clone()

    return W_current, mask, global_perm, best_time_relative, history, after_tetris_index_in_history

def random_swaps_find_mask(W, block_size=(16, 1), sparsity=0.5, max_iter=10, verbose=True):
    t0 = time.perf_counter()
    best_time_relative = 0.0
    history = []

    W_current = W.clone()
    permutation = torch.arange(W_current.shape[1], device=W.device)
    
    _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
    original_pruned = W_current.abs()[mask == 0].sum().item()
    history.append((best_time_relative, original_pruned))
    
    if verbose:
        print(f"{'FIRST MASK':<{PRINT_C}}{original_pruned:<{PRINT_C}.10f}")
    previous_swap = original_pruned

    if verbose:
        print(f"{'AFTER_SWAP':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")
        
    for iteration_num in range(max_iter):
        if (iteration_num + 1) % 10 == 0:
            print(f"Random swap iteration {iteration_num + 1}/{max_iter}")

        previous_mask = mask.clone()
        previous_permutation = permutation.clone()
        previous_W = W_current.clone()

        # Random swaps
        for _ in range(10):
            # Grab two random indices
            indices = torch.randperm(len(permutation), device=W.device)[:2]
            i, j = indices[0], indices[1]
            
            permutation[[i, j]] = permutation[[j, i]]
            W_current[:, [i, j]] = W_current[:, [j, i]]

        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_swap = W_current.abs()[mask == 0].sum().item()
        
        if previous_swap - after_swap > TOLERANCE:
            if verbose:
                print(f"{after_swap:<{PRINT_C}.10f}{previous_swap-after_swap:<{PRINT_C}.10f}{(previous_swap-after_swap)/previous_swap*100:<{PRINT_C}.10f}{(original_pruned-after_swap)/original_pruned*100:<{PRINT_C}.10f}")
            previous_swap = after_swap
            best_time_relative = time.perf_counter() - t0
            history.append((best_time_relative, after_swap))
        else:
            mask = previous_mask.clone()
            permutation = previous_permutation.clone()
            W_current = previous_W.clone()

    return W_current, mask, permutation, best_time_relative, history


def sort_columns_by_norm(W, block_size=(1, 16), sparsity=0.5, verbose=True):
    W_current = W.clone()
    _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
    original_pruned = W_current.abs()[mask == 0].sum().item()
    
    # 1. Calculate column norms
    col_norms = W.abs().sum(dim=0)

    # 2. Get permutation that sorts columns by norm
    # It should be descending order, because it the block pruning function we fit the blocks from left to right and 
    # everything at the end that doesnt fit into a full block is pruned
    sorted_permutation = torch.argsort(col_norms, descending=True)
    
    # 3. Apply permutation
    W_sorted = W[:, sorted_permutation]

    # 4. Apply block sparsity pruning
    _, mask = block_sparsity_pruning(W_sorted, block_size, sparsity)
    
    after_mask = W_sorted.abs()[mask == 0].sum().item()
    if verbose:
        print(f"{'ORIGINAL':<{PRINT_C}}{'AFTER SORT':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}")
        print(f"{original_pruned:<{PRINT_C}.10f}{after_mask:<{PRINT_C}.10f}{original_pruned-after_mask:<{PRINT_C}.10f}{(original_pruned-after_mask)/original_pruned*100:<{PRINT_C}.10f}")
    
    return W_sorted, mask, sorted_permutation


def random_permutation_pruning(W, block_size=(1, 16), sparsity=0.5, verbose=True):
    W_current = W.clone()
    _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
    original_pruned = W_current.abs()[mask == 0].sum().item()
    
    # 1. Generate a completely random permutation of column indices
    random_permutation = torch.randperm(W.shape[1], device=W.device)

    # 2. Apply permutation
    W_random = W[:, random_permutation]
    
    # 3. Apply block sparsity pruning
    _, mask = block_sparsity_pruning(W_random, block_size, sparsity)
    
    after_mask = W_random.abs()[mask == 0].sum().item()
    if verbose:
        print(f"{'ORIGINAL':<{PRINT_C}}{'AFTER RAND':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}")
        print(f"{original_pruned:<{PRINT_C}.10f}{after_mask:<{PRINT_C}.10f}{original_pruned-after_mask:<{PRINT_C}.10f}{(original_pruned-after_mask)/original_pruned*100:<{PRINT_C}.10f}")
    
    return W_random, mask, random_permutation

if __name__ == "__main__":
    # original = torch.load("xy.pt").detach().numpy()
    original = torch.load("xy.pt").cpu().detach().numpy()
    # original = original[:200, :2000]
    # print("ORIGINAL:", original, sep="\n")
    # _, original_mask = block_sparsity_pruning(original, block_size=(1, 16))

    BLOCK_SIZE = (1, 8)
    SPARSITY = 0.5
    MAX_ITER = 100
    RANDOM_SWAPS = 500

    print(
        f"BLOCK SIZE: {BLOCK_SIZE}, SPARSITY: {SPARSITY}, MAX ITER: {MAX_ITER}, SHAPE: {original.shape}")

    # Apply original tetris
    reordered, final_mask, permutation = original_tetris_pruning(
        original, block_size=BLOCK_SIZE, sparsity=SPARSITY, max_iter=MAX_ITER)

    # # Apply OUR algorithm
    # reordered, final_mask, permutation = tetris_pruning(
    #     original, block_size=BLOCK_SIZE, sparsity=SPARSITY, max_iter=MAX_ITER, random_swaps=RANDOM_SWAPS)


    # # Apply random swaps
    # random_swaps_find_mask(
    #     original, block_size=BLOCK_SIZE, sparsity=SPARSITY, max_iter=MAX_ITER)
