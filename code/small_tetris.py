from tetris import calculate_row_gains_numpy, find_optimal_permutation
import numpy as np

def small_tetris(W, mask):
    # 3. Calculate gains using inverted mask
    inverted_mask = 1 - mask
    G = calculate_row_gains_numpy(W, inverted_mask)
    print("G\n", G)

    # 4. Find optimal permutation
    permutation = find_optimal_permutation(G)
    print(permutation)

    # 5. Apply permutation
    W = W[permutation, :]
    
    return W

print("HAHAHAHAH")

# SMALL TETRIS 
mat = np.arange(1, 19).reshape(3, 6)
mask = np.ones((3, 6))
maskk = np.random.choice([True, False], size=mat.shape)
mat[maskk] *= -1
mask[1] = 0
mask[2, 3] = 0
print(mask)
    
small_tet = small_tetris(mat, mask)
print(small_tet)
