# pylint: disable=missing-docstring
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

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
def calculate_exchange_gains(W, M):
    # S[i,j] = sum(|W[:,i]| * M[:,j])
    S = np.abs(W) @ M.T

    # Get diagonal elements
    L = np.diag(S)

    # Calculate gain matrix
    G = L[:, None] + L[None, :] - S - S.T

    return G

def tetris_pruning(W, block_size=(16,1), sparsity=0.5, max_iter=10):
    W_current = W.copy()
    print("BLOCK\t\tTETRIS\t\tDIFF")
    
    for _ in range(max_iter):
         # 1. Apply pruning to get mask
        _, mask = block_sparsity_pruning(W_current, block_size, sparsity)
        after_block = np.abs(W_current)[mask == 0].sum()

        # 2. Invert mask to match paper's format (1 = pruned, 0 = kept)
        inverted_mask = 1 - mask

        # 3. Calculate gains using inverted mask
        G = calculate_exchange_gains(W_current, inverted_mask)

        # 4. Find optimal permutation
        permutation = find_optimal_permutation(G)
        # print(permutation)

        # 5. Apply permutation
        W_current = W_current[permutation, :]
        after_tetris = np.abs(W_current)[mask == 0].sum()
        print(f"{after_block:.10f}\t{after_tetris:10f}\t{after_block-after_tetris:10f}")

    return W_current, mask

def small_tetris(W, mask):
    # 3. Calculate gains using inverted mask
    inverted_mask = 1 - mask
    G = calculate_exchange_gains(W, inverted_mask)

    # 4. Find optimal permutation
    permutation = find_optimal_permutation(G)
    print(permutation)

    # 5. Apply permutation
    W = W[permutation, :]
    
    return W

# SMALL TETRIS 
mat = np.arange(1, 19).reshape(3, 6)
mask = np.ones((3, 6))
mask[1] = 0

small_tet = small_tetris(mat, mask)
print(small_tet)


# ## NORMAL TETRIS
# original = torch.load("xy.pt").detach().numpy()
# _, original_mask = block_sparsity_pruning(original, block_size=(16, 1))

# # Apply reordering and pruning
# reordered, final_mask = tetris_pruning(original, block_size=(16, 1))

# print("AVG NUMBER", np.percentile(np.abs(original), 1))
# print("PRUNED FINAL:", reordered*final_mask)
# print("PRUNED SHAPE:", reordered.shape, "ORGINAL SHAPE:", original.shape)
# print(f"After tetris pruned sum: {np.abs(reordered)[final_mask == 0].sum():.10f}")
