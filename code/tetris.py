# pylint: disable=missing-docstring
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

PRINT_C = 25


# Not using, just to check if vectoried version is correct when I am paranoid
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
               original = abs(W[row,i]) * M[row,i] + abs(W[row,j]) * M[row,j]
               
               # Score after swapping columns
               swapped = abs(W[row,i]) * M[row,j] + abs(W[row,j]) * M[row,i]
               
               # Add difference to total gain
               gain += original - swapped
               
           gains[i,j] = gain
           
   return gains

def block_sparsity_pruning(W, block_size=(16,1), sparsity=0.5):
    rows, cols = W.shape
    block_rows, block_cols = block_size

    # Calculate number of blocks
    n_blocks_row = rows // block_rows
    n_blocks_col = cols // block_cols

    # Reshape into blocks to compute L1 norms
    blocks = W[:n_blocks_row*block_rows, :n_blocks_col*block_cols]
    blocks = blocks.reshape(n_blocks_row, block_rows, n_blocks_col, block_cols)

    # Compute L1 norm for each block
    block_norms = np.abs(blocks).sum(axis=(1,3))

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
    permutation = np.arange(W.shape[1])
    max_item = 10
    ii = 0
    while max_item > 1e-4:
        if(ii % 100 == 0):
            print(f"ITERATION {ii} MAX ITEM: {max_item:.10f}")
        ii += 1
        G = calculate_column_gains_numpy(W, M)
        max_item = np.max(G)
        i, j = np.unravel_index(np.argmax(G), G.shape)
        # print(f"MAX ITEM: {max_item:.10f} AT {i}, {j}")
        W[:, [i, j]] = W[:, [j, i]]
        permutation[[i, j]] = permutation[[j, i]]
        
    np.set_printoptions(threshold=np.inf)
    print(permutation)
    return permutation

def original_tetris_pruning(W, block_size=(16,1), sparsity=0.5, max_iter=1):
    W_current = W.copy()
    original_pruned = 0
    print(f"{'BLOCK':<{PRINT_C}}{'ORIG TETRIS':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")
    
    for _ in range(max_iter):
        # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = np.abs(W_current)[mask == 0].sum()
        if original_pruned == 0:
            original_pruned = after_block

        # 2. Find optimal permutation
        permutation = original_tetris_find_optimal_permutation(W_current, mask)

        # 3. Apply permutation
        W_current = W_current[:, permutation]
        after_tetris = np.abs(W_current)[mask == 0].sum()
        print(f"{after_block:<{PRINT_C}.10f}{after_tetris:<{PRINT_C}.10f}{after_block-after_tetris:<{PRINT_C}.10f}{(after_block-after_tetris)/after_block*100:<{PRINT_C}.10f}{(original_pruned-after_tetris)/original_pruned*100:<{PRINT_C}.10f}")

    return W_current, mask
    

def tetris_pruning(W, block_size=(16,1), sparsity=0.5, max_iter=1):
    W_current = W.copy()
    original_pruned = 0
    print(f"{'BLOCK':<{PRINT_C}}{'TETRIS':<{PRINT_C}}{'DIFF':<{PRINT_C}}{'DIFF %':<{PRINT_C}}{'TOTAL DIFF %':<{PRINT_C}}")
    
    for _ in range(max_iter):
         # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = np.abs(W_current)[mask == 0].sum()
        if original_pruned == 0:
            original_pruned = after_block

        # 2. Invert mask to match paper's format (1 = pruned, 0 = kept)
        inverted_mask = 1 - mask

        # 3. Calculate gains using inverted mask
        G = calculate_column_gains_numpy(W_current, inverted_mask)

        # 4. Find optimal permutation
        permutation = find_optimal_permutation(G)
        # np.set_printoptions(threshold=np.inf)
        # print(permutation)

        # 5. Apply permutation
        W_current = W_current[:, permutation]
        after_tetris = np.abs(W_current)[mask == 0].sum()
        print(f"{after_block:<{PRINT_C}.10f}{after_tetris:<{PRINT_C}.10f}{after_block-after_tetris:<{PRINT_C}.10f}{(after_block-after_tetris)/after_block*100:<{PRINT_C}.10f}{(original_pruned-after_tetris)/original_pruned*100:<{PRINT_C}.10f}")

    return W_current, mask

if __name__ == "__main__":
    original = torch.load("xy.pt").detach().numpy()
    # original = original[:64, :128]
    print(original.shape)
    # print("ORIGINAL:", original, sep="\n")
    # _, original_mask = block_sparsity_pruning(original, block_size=(1, 16))

    # Apply reordering and pruning
    reordered, final_mask = tetris_pruning(original, block_size=(1, 16))

    # Apply original tetris
    # reordered, final_mask = original_tetris_pruning(original, block_size=(1, 16))

    
    # print("AVG NUMBER", np.percentile(np.abs(original), 1))
    # print("PRUNED FINAL:", reordered*final_mask, sep="\n")
    # print("PRUNED SHAPE:", reordered.shape, "ORGINAL SHAPE:", original.shape)
    # print(f"After tetris pruned sum: {np.abs(reordered)[final_mask == 0].sum():.10f}")
