import papermill as pm
import scrapbook as sb
import os

# Define parameters for each run
experiments = [
    {'ALGORITHM': 'our_tetris', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'NOISE_SCALE': 0, 'MAX_ITER': 10, 'RANDOM_SWAPS': 100, 'SPARSITY': 0.5},
    # {'ALGORITHM': 'our_tetris', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'NOISE_SCALE': 5, 'MAX_ITER': 10, 'RANDOM_SWAPS': 1000, 'SPARSITY': 0.5},
    # {'ALGORITHM': 'sort_columns_by_norm', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 4, 'NOISE_SCALE': 5, 'MAX_ITER': 10, 'RANDOM_SWAPS': 10, 'SPARSITY': 0.5},
    # {'ALGORITHM': 'sort_columns_by_norm', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 8, 'NOISE_SCALE': 5, 'MAX_ITER': 10, 'RANDOM_SWAPS': 10, 'SPARSITY': 0.5},
    # {'ALGORITHM': 'block_wanda', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'NOISE_SCALE': 5, 'MAX_ITER': 10, 'RANDOM_SWAPS': 10, 'SPARSITY': 0.5},
    # {'ALGORITHM': 'our_tetris', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'NOISE_SCALE': 5, 'MAX_ITER': 10, 'RANDOM_SWAPS': 10, 'SPARSITY': 0.5},
    # {'ALGORITHM': 'original_tetris', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'NOISE_SCALE': 5, 'MAX_ITER': 10, 'RANDOM_SWAPS': 10, 'SPARSITY': 0.5},
    # {'ALGORITHM': 'random_permutation_pruning', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'NOISE_SCALE': 5, 'MAX_ITER': 10, 'RANDOM_SWAPS': 10, 'SPARSITY': 0.5},
]

OUTPUT_FOLDER = "experiment_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
summary_file = os.path.join(OUTPUT_FOLDER, "results_summary.md")

if not os.path.exists(summary_file):
    with open(summary_file, "w") as f:
        f.write("| Algorithm | Block Size | Sparsity | Max Iter | Noise Scale | Random Swaps | Perplexity | Execution Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
else:
    with open(summary_file, "a") as f:
        f.write("|---|---|---|---|---|---|---|---|\n")


for exp in experiments:
    alg = exp['ALGORITHM']
    r = exp['BLOCK_ROWS']
    c = exp['BLOCK_COLS']
    iters = exp['MAX_ITER']
    swaps = exp['RANDOM_SWAPS']
    sparsity = exp['SPARSITY']
    noise_scale = exp['NOISE_SCALE']
    
    file_name = f"results_{alg}_{r}x{c}_noise_{noise_scale}_iters_{iters}_swaps_{swaps}_sparsity_{sparsity}.ipynb"
    output_path = os.path.join(OUTPUT_FOLDER, file_name)
    
    print(f"Starting {file_name}...")
    
    try:
        pm.execute_notebook(
            'Smollm-2-145M-pruning-and-eval.ipynb',
            output_path,
            parameters=exp
        )
        
        # Extract data from the executed notebook using scrapbook
        nb = sb.read_notebook(output_path)
        
        # Extract Perplexity
        try:
            perp = nb.scraps['perplexity'].data
            perp_str = f"{float(perp):.4f}"
        except KeyError:
            perp_str = "FAILED (No scrap)"
            
        # Extract Execution Time
        try:
            time_val = nb.scraps['execution_time'].data
            time_str = f"{float(time_val):.2f}"
        except KeyError:
            time_str = "N/A"
            
    except Exception as e:
        print(f"Error running {file_name}: {e}")
        perp_str = "CRASHED"
        time_str = "CRASHED"
    
    # Log results to summary file
    with open(summary_file, "a") as f:
        f.write(f"| {alg} | {r}x{c} | {sparsity} | {iters} | {noise_scale} | {swaps} | **{perp_str}** | {time_str} |\n")
        
    print(f"Finished {file_name}! Perplexity: {perp_str} | Time: {time_str}s\n")
