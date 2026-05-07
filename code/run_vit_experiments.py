"""Run ViT pruning experiments via Papermill, in the same shape as run_experiments.py.

Each experiment runs ViT-pruning-and-eval.ipynb with parameters and collects:
  baseline_accuracy, accuracy_single_layer, accuracy_all_layers,
  relative_errors (list per layer), execution_time, single_layer_name.

Edit the `experiments` list to control what gets run. The big sweep loop at
the top is preserved from the SmolLM script for reference; the assignment
below it overrides with a small set you actually want to run.
"""
import os
import papermill as pm
import scrapbook as sb


# ---------------------------------------------------------------------------
# 1. Build the full sweep (you can keep this as a reference / generate-all)
# ---------------------------------------------------------------------------

max_iters = [1, 5, 10]
random_swaps = [0, 10, 100]
sparsities = [0.3, 0.4, 0.5, 0.6]
block_shapes = [(1, 2), (1, 4), (1, 8)]
swap_fractions = [1/20, 1/30, 1/40]
algorithms = [
    'block_only', 'sort_columns_by_norm', 'random_swaps_find_mask',
    'original_tetris', 'our_tetris',
]

experiments = []

# Always include a no-prune baseline for accuracy reference
experiments.append({'ALGORITHM': 'no_prune', 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5})

for block_shape in block_shapes:
    for sparsity in sparsities:
        for alg in algorithms:
            base = {
                'ALGORITHM': alg,
                'BLOCK_ROWS': block_shape[0],
                'BLOCK_COLS': block_shape[1],
                'SPARSITY': sparsity,
            }
            if alg in ('block_only', 'sort_columns_by_norm'):
                experiments.append(base)
                continue
            for iters in max_iters:
                if alg == 'random_swaps_find_mask':
                    for swap_fraction in swap_fractions:
                        experiments.append({
                            **base, 'MAX_ITER': iters,
                            'SWAP_FRACTION': swap_fraction, 'SORT_START': False,
                        })
                        experiments.append({
                            **base, 'MAX_ITER': iters,
                            'SWAP_FRACTION': swap_fraction, 'SORT_START': True,
                        })
                    continue
                if alg == 'original_tetris':
                    experiments.append({**base, 'MAX_ITER': iters})
                    continue
                # our_tetris
                for swaps in random_swaps:
                    experiments.append({**base, 'MAX_ITER': iters, 'RANDOM_SWAPS': swaps})

print(f"Total experiments in full sweep: {len(experiments)}")

# ---------------------------------------------------------------------------
# 2. Override with the small set you actually want to run right now
#    (comment this block out to run the full sweep)
# ---------------------------------------------------------------------------

experiments = [
    {'ALGORITHM': 'no_prune',                'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
    {'ALGORITHM': 'sort_columns_by_norm',    'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
    {'ALGORITHM': 'random_swaps_find_mask',  'MAX_ITER': 10, 'SWAP_FRACTION': 1/30,
     'SORT_START': False, 'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
    {'ALGORITHM': 'random_swaps_find_mask',  'MAX_ITER': 10, 'SWAP_FRACTION': 1/30,
     'SORT_START': True,  'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
    {'ALGORITHM': 'original_tetris',         'MAX_ITER': 10,
     'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
    {'ALGORITHM': 'our_tetris',              'MAX_ITER': 10, 'RANDOM_SWAPS': 10,
     'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5},
]

# Set this once and it'll apply to every experiment that doesn't already override.
DEFAULT_N_CALIB_SAMPLES = 128
for exp in experiments:
    exp.setdefault('N_CALIB_SAMPLES', DEFAULT_N_CALIB_SAMPLES)


# ---------------------------------------------------------------------------
# 3. Output setup
# ---------------------------------------------------------------------------

OUTPUT_FOLDER = "vit_experiment_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
summary_file = os.path.join(OUTPUT_FOLDER, "results_summary.md")

HEADER = (
    "| Algorithm | Max Iter | Random Swaps | Swap Fraction | Sparsity | Block Shape "
    "| Avg Rel Error | Acc (single layer) | Acc (all layers) | Acc (baseline) "
    "| Single Layer | Time (s) |\n"
)
DIVIDER = "|" + "|".join(["---"] * 11) + "|\n"

if not os.path.exists(summary_file):
    with open(summary_file, "w") as f:
        f.write(HEADER)
        f.write(DIVIDER)
else:
    with open(summary_file, "a") as f:
        f.write(DIVIDER)


# ---------------------------------------------------------------------------
# 4. Run each experiment
# ---------------------------------------------------------------------------

for exp in experiments:
    alg = exp['ALGORITHM']
    if alg == 'random_swaps_find_mask' and exp.get('SORT_START', False):
        alg_label = alg + '_sort_start'
    else:
        alg_label = alg

    iters = exp.get('MAX_ITER', 'na')
    swaps = exp.get('RANDOM_SWAPS', 'na')
    swap_fraction = exp.get('SWAP_FRACTION', 'na')
    if swap_fraction != 'na':
        swap_fraction_str = f"{swap_fraction:.4f}"
    else:
        swap_fraction_str = 'na'

    r = exp['BLOCK_ROWS']
    c = exp['BLOCK_COLS']
    sparsity = exp['SPARSITY']

    file_name = (
        f"{alg_label}_{r}x{c}"
        f"_iters_{iters}"
        f"_swaps_{swaps}"
        f"_swapfrac_{swap_fraction_str}"
        f"_sparsity_{sparsity}.ipynb"
    )
    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    print(f"Starting {file_name}...")

    baseline_acc_str = "N/A"
    acc_single_str = "N/A"
    acc_all_str = "N/A"
    avg_error_str = "N/A"
    time_str = "N/A"
    single_layer_name = "N/A"

    try:
        pm.execute_notebook(
            'ViT-pruning-and-eval.ipynb',
            output_path,
            parameters=exp,
        )

        nb = sb.read_notebook(output_path)

        def grab(key, fmt=None):
            try:
                v = nb.scraps[key].data
                return f"{float(v):{fmt}}" if fmt else v
            except KeyError:
                return None

        baseline_v = grab('baseline_accuracy', '.4f')
        if baseline_v is not None:
            baseline_acc_str = baseline_v

        acc_single_v = grab('accuracy_single_layer', '.4f')
        if acc_single_v is not None:
            acc_single_str = acc_single_v

        acc_all_v = grab('accuracy_all_layers', '.4f')
        if acc_all_v is not None:
            acc_all_str = acc_all_v

        time_v = grab('execution_time', '.2f')
        if time_v is not None:
            time_str = time_v

        single_layer_name_v = grab('single_layer_name')
        if single_layer_name_v is not None:
            single_layer_name = str(single_layer_name_v)

        try:
            errors_list = nb.scraps['relative_errors'].data
            avg_error = sum(errors_list) / len(errors_list)
            avg_error_str = f"{avg_error:.6f}"
        except KeyError:
            avg_error_str = "FAILED (No scrap)"

    except Exception as e:
        print(f"Error running {file_name}: {e}")
        baseline_acc_str = "CRASHED"
        acc_single_str = "CRASHED"
        acc_all_str = "CRASHED"
        avg_error_str = "CRASHED"
        time_str = "CRASHED"

    with open(summary_file, "a") as f:
        f.write(
            f"| {alg_label} | {iters} | {swaps} | {swap_fraction_str} | {sparsity} | ({r}x{c}) "
            f"| {avg_error_str} | **{acc_single_str}** | **{acc_all_str}** | {baseline_acc_str} "
            f"| `{single_layer_name}` | {time_str} |\n"
        )

    print(
        f"Finished {file_name}!  "
        f"single={acc_single_str}  all={acc_all_str}  baseline={baseline_acc_str}  "
        f"time={time_str}s"
    )
