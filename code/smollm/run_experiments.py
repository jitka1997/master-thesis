import random

import papermill as pm
import scrapbook as sb
import os

# Define parameters for each run
max_iters = [1, 3, 5]
random_swaps = [0, 10, 100]
sparsities = [0.3, 0.4, 0.5, 0.6]
block_shapes = [(1, 2), (1, 4), (1, 8)]
swap_fractions = [1/20, 1/30, 1/40]
algorithms = ['block_wanda', 'sort_columns_by_norm', 'random_swaps', 'original_tetris', 'our_tetris']

experiments = []

# dont forget no_prune special case
for block_shape in block_shapes:
    for sparsity in sparsities:
        for alg in algorithms:
            if alg == 'block_wanda' or alg == 'sort_columns_by_norm':
                experiments.append({'ALGORITHM': alg, 'BLOCK_ROWS': block_shape[0], 'BLOCK_COLS': block_shape[1], 'SPARSITY': sparsity})
                continue
            for iters in max_iters:
                if alg == 'random_swaps':
                    for swap_fraction in swap_fractions:
                        experiments.append({'ALGORITHM': alg, 'MAX_ITER': iters, 'SWAP_FRACTION': swap_fraction, 'SORT_START': False, 'BLOCK_ROWS': block_shape[0], 'BLOCK_COLS': block_shape[1], 'SPARSITY': sparsity})
                        experiments.append({'ALGORITHM': alg, 'MAX_ITER': iters, 'SWAP_FRACTION': swap_fraction, 'SORT_START': True, 'BLOCK_ROWS': block_shape[0], 'BLOCK_COLS': block_shape[1], 'SPARSITY': sparsity})
                    continue
                if alg == 'original_tetris':
                    experiments.append({'ALGORITHM': alg, 'MAX_ITER': iters, 'BLOCK_ROWS': block_shape[0], 'BLOCK_COLS': block_shape[1], 'SPARSITY': sparsity})
                    continue
                for swaps in random_swaps:
                    experiments.append({'ALGORITHM': alg, 'MAX_ITER': iters, 'RANDOM_SWAPS': swaps, 'BLOCK_ROWS': block_shape[0], 'BLOCK_COLS': block_shape[1], 'SPARSITY': sparsity})

print(f"Total experiments to run: {len(experiments)}")
# for exp in experiments:
#     print(exp)

# # Use this part to run a few experiments
# experiments = [
#     {'ALGORITHM': 'sort_columns_by_norm', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
#     {'ALGORITHM': 'random_swaps', 'MAX_ITER': 10, 'SWAP_FRACTION': 1/30, 
#      'SORT_START': False, 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
#     {'ALGORITHM': 'random_swaps', 'MAX_ITER': 10, 'SWAP_FRACTION': 1/30, 
#      'SORT_START': True, 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
# ]

# experiments = []

# seeds = [random.randint(0, 1000) for _ in range(50)]
# for i in range(50):
#     experiments.append({
#         'ALGORITHM': 'sort_columns_by_norm', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5, 'SEED': seeds[i]
#     })
#     experiments.append({
#         'ALGORITHM': 'random_swaps', 'MAX_ITER': 10, 'SWAP_FRACTION': 1/30, 
#         'SORT_START': False, 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5, 'SEED': seeds[i]
#     })
#     experiments.append({
#         'ALGORITHM': 'random_swaps', 'MAX_ITER': 10, 'SWAP_FRACTION': 1/30, 
#         'SORT_START': True, 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5, 'SEED': seeds[i]
#     })

# experiments = []

# BLOCK_ROWS = 1
# BLOCK_COLS = 2
# SPARSITY = 0.5

# common = {
#     'BLOCK_ROWS': BLOCK_ROWS,
#     'BLOCK_COLS': BLOCK_COLS,
#     'SPARSITY': SPARSITY,
# }

# experiments.append({**common, 'ALGORITHM': 'block_wanda'})
# experiments.append({**common, 'ALGORITHM': 'sort_columns_by_norm'})
# experiments.append({**common, 'ALGORITHM': 'random_permutation_pruning'})
# experiments.append({**common, 'ALGORITHM': 'original_tetris', 'MAX_ITER': 5})
# experiments.append({**common, 'ALGORITHM': 'our_tetris', 'MAX_ITER': 5, 'RANDOM_SWAPS': 10})
# experiments.append({**common, 'ALGORITHM': 'random_swaps',
#                     'MAX_ITER': 10, 'SWAP_FRACTION': 1/30, 'SORT_START': False})
# experiments.append({**common, 'ALGORITHM': 'random_swaps',
#                     'MAX_ITER': 10, 'SWAP_FRACTION': 1/30, 'SORT_START': True})
# 
# print(f"Total experiments to run: {len(experiments)}")


OUTPUT_FOLDER = "experiment_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
summary_file = os.path.join(OUTPUT_FOLDER, "results_summary.md")

# Write headers if new, or append a divider if it already exists
if not os.path.exists(summary_file):
    with open(summary_file, "w") as f:
        f.write("| Algorithm | Max Iter | Random Swaps | Swap Fraction | Sparsity | Block Shape | Avg Rel Error | Perplexity | Execution Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
else:
    with open(summary_file, "a") as f:
        # Added the 7-column divider back for new batch runs!
        f.write("|---|---|---|---|---|---|---|---|---|\n")

for exp in experiments:
    alg = exp['ALGORITHM']
    if alg == 'random_swaps' and exp.get('SORT_START', False):
        alg = alg + "_sort_start"

    iters = exp.get('MAX_ITER', 'na')
    swaps = exp.get('RANDOM_SWAPS', 'na')
    swap_fraction = exp.get('SWAP_FRACTION', 'na')
    if swap_fraction != 'na':
        swap_fraction = f"{swap_fraction:.4f}"

    r = exp['BLOCK_ROWS']
    c = exp['BLOCK_COLS']
    sparsity = exp['SPARSITY']

    file_name = f"{alg}_{r}x{c}_iters_{iters}_swaps_{swaps}_swapfrac_{swap_fraction}_sparsity_{sparsity}.ipynb"
    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    print(f"Starting {file_name}...")

    try:
        pm.execute_notebook(
            'Smollm-2-pruning-and-eval.ipynb',
            output_path,
            parameters=exp,
        )

        nb = sb.read_notebook(output_path)

        try:
            perp = nb.scraps['perplexity'].data
            perp_str = f"{float(perp):.4f}"
        except KeyError:
            perp_str = "FAILED (No scrap)"

        try:
            time_val = nb.scraps['execution_time'].data
            time_str = f"{float(time_val):.2f}"
        except KeyError:
            time_str = "N/A"

        try:
            errors_list = nb.scraps['relative_errors'].data
            avg_error = sum(errors_list) / len(errors_list)
            avg_error_str = f"{avg_error:.4f}"
        except KeyError:
            avg_error_str = "FAILED (No scrap)"

    except Exception as e:
        print(f"Error running {file_name}: {e}")
        perp_str = "CRASHED"
        time_str = "CRASHED"
        avg_error_str = "CRASHED"

    with open(summary_file, "a") as f:
        f.write(f"| {alg} | {iters} | {swaps} | {swap_fraction} | {sparsity} | ({r}x{c}) | {avg_error_str} | **{perp_str}** | {time_str} |\n")

    print(f"Finished {file_name}! Perplexity: {perp_str} | Time: {time_str}s | seed: {exp.get('SEED', 'N/A')}")
