# pylint: disable=missing-docstring
import time
import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

PRINT_C = 25
TOLERANCE = 1e-4


def block_sparsity_pruning(W, block_size=(1, 2), sparsity=0.5):
    rows, cols = W.shape
    block_rows, block_cols = block_size

    n_blocks_row = rows // block_rows
    n_blocks_col = cols // block_cols

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


def find_optimal_permutation_exact(W, M_pruned):
    # C[c, p] = sum_i |W[i, c]| * M_pruned[i, p]
    C = torch.matmul(W.abs().T, M_pruned.float())
    C_cpu = C.cpu().numpy()
    
    # Minimize total pruned mass
    row_ind, col_ind = linear_sum_assignment(C_cpu, maximize=False)
    # row_ind is 0..n-1 (columns), col_ind[c] is the position where column c goes.
    # We want the permutation pi such that W[:, pi] gives the rearranged matrix,
    # i.e., pi[p] = the original column that ends up at position p.
    # Since col_ind[c] = destination of column c, pi is the inverse:
    pi = np.argsort(col_ind)
    return torch.tensor(pi, device=W.device)

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

def original_tetris_pruning(W, block_size=(1, 2), sparsity=0.5, max_iter=10, verbose=True):
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


def tetris_pruning(W, block_size=(1, 2), sparsity=0.5, max_iter=10, random_swaps=20, inner_refine=2, verbose=True):
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
        print(f"{'BLOCK':<{PRINT_C}}{'TETRIS':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")

    for iteration_num in range(max_iter):
        if (iteration_num + 1) % 10 == 0 and verbose:
            print(f"Tetris iteration {iteration_num + 1}/{max_iter}")

        # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = W_current.abs()[mask == 0].sum().item()

        # 2. Invert mask (1 = pruned, 0 = kept) to calculate pruned mass easily
        inverted_mask = 1.0 - mask

        # 3. Add noise
        starting_noise = 25
        progress = iteration_num / (max_iter)
        W_noisy = add_noise(W_current, starting_noise - progress * starting_noise, distribution='normal')

        # 4. Find optimal permutation
        permutation = find_optimal_permutation_exact(W_noisy, inverted_mask)


        global_perm = global_perm[permutation]

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

        previous_global_perm = global_perm.clone()
        previous_W = W_current.clone()
        
        # Random swaps
        for _ in range(5):
            # Get a 1D tensor containing 2 random indices
            indices = torch.randperm(W_current.shape[1], device=W.device)[:2]
            
            # Get the reversed version of those indices
            rev_indices = torch.flip(indices, dims=[0])
            
            global_perm[indices] = global_perm[rev_indices]
            W_current[:, indices] = W_current[:, rev_indices]

        # Optimal permutation
        for _ in range(inner_refine):
            _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
            inverted_mask = 1.0 - mask
            permutation = find_optimal_permutation_exact(W_current, inverted_mask)
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
            global_perm = previous_global_perm.clone()

    # Sanity check: global_perm should reproduce W_current from the original W
    if (not torch.allclose(W[:, global_perm], W_current)):
        print("Warning: global_perm does not reproduce W_current from original W. This may indicate a bug.")

    return W_current, mask, global_perm, best_time_relative, history, after_tetris_index_in_history

def random_swaps(W, block_size=(1, 2), sparsity=0.5, max_iter=10, sort_start=False, swap_fraction=1/30, verbose=True):
    t0 = time.perf_counter()
    best_time_relative = 0.0
    history = []

    W_current = W.clone()
    permutation = torch.arange(W_current.shape[1], device=W.device)

    # Add sorting by column norm at the start
    if sort_start:
        col_norms = W_current.abs().sum(dim=0)
        sorted_permutation = torch.argsort(col_norms, descending=True)
        W_current = W_current[:, sorted_permutation]
        permutation = sorted_permutation.clone()

    
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
        for _ in range(max(1, int(swap_fraction * len(permutation)))): 
            # Get a 1D tensor containing 2 random indices
            indices = torch.randperm(len(permutation), device=W.device)[:2]
            
            # Get the reversed version of those indices
            rev_indices = torch.flip(indices, dims=[0])
            
            permutation[indices] = permutation[rev_indices]
            W_current[:, indices] = W_current[:, rev_indices]

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


def sort_columns_by_norm(W, block_size=(1, 2), sparsity=0.5, verbose=True):
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
