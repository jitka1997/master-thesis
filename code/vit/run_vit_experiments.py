"""Run ViT pruning experiments via Papermill.

Each experiment runs ViT-pruning-and-eval.ipynb with parameters and collects:
  baseline_accuracy, accuracy_single_layer, accuracy_all_layers,
  relative_errors (list per layer), execution_time, single_layer_name,
  whole_model_validation_time.

Edit the `experiments` list to control what gets run.
"""
import os
import papermill as pm
import scrapbook as sb


# Very small model
MODEL_NAME = "test_vit3.r160_in1k"

# Bigger model
# MODEL_NAME = "vit_wee_patch16_reg1_gap_256.sbb_in1k"


# ---------------------------------------------------------------------------
# 1. Define experiments to run. Defaults for every experiment are set in section 2
# ---------------------------------------------------------------------------

# algorithm options:
# 'original_tetris', 'our_tetris', 'random_swaps', 'sort_columns_by_norm', 'block_wanda', 'random_swaps'


# ---------------------------------------------------------------------------
# Example single experiment
experiments = [
    {
        'ALGORITHM': 'block_wanda',
        'BLOCK_ROWS': 1,
        'BLOCK_COLS': 2,
        'SPARSITY': 0.5,
        'SKIP_ACCURACY': True,
        'MODEL_NAME': MODEL_NAME,
    },
]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# # Original tetris iterations hyperparameter sweep

# experiments = []
# DEFAULTS = {'BLOCK_ROWS': 1, 'BLOCK_COLS': 8, 'SPARSITY': 0.5, 'SKIP_ACCURACY': True}

# # MAX_ITER sweep
# for mi in [1, 3, 5, 10, 20, 50, 100]:
#     experiments.append({**DEFAULTS, 'ALGORITHM': 'original_tetris',
#                         'MAX_ITER': mi})
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# # Full grid sweep on the main 3 hyperparameters (MAX_ITER, RANDOM_SWAPS, INNER_REFINE)

# experiments = []
# for mi in [1, 3, 5, 10, 20]:
#     for rs in [0, 10, 20, 50, 100]:
#         for ir in [1, 2, 3]:
#             experiments.append({
#                 'ALGORITHM': 'our_tetris',
#                 'BLOCK_ROWS': 1, 'BLOCK_COLS': 8,
#                 'SPARSITY': 0.5, 'SKIP_ACCURACY': True,
#                 'MAX_ITER': mi, 'RANDOM_SWAPS': rs, 'INNER_REFINE': ir,
#             })

# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# # Original and our tetris max iter = 5, inner refine = 2, random swaps = 100
# experiments = []
# for bc in [2, 4, 8, 16, 32]:
#     experiments.append({
#         'ALGORITHM': 'original_tetris',
#         'MAX_ITER': 5,
#         'BLOCK_ROWS': 1, 'BLOCK_COLS': bc, 'SPARSITY': 0.5,
#         'SKIP_ACCURACY': False,
#     })

# for bc in [2, 4, 8, 16, 32]:
#     experiments.append({
#         'ALGORITHM': 'our_tetris',
#         'MAX_ITER': 5, 'RANDOM_SWAPS': 100, 'INNER_REFINE': 2,
#         'BLOCK_ROWS': 1, 'BLOCK_COLS': bc, 'SPARSITY': 0.5,
#         'SKIP_ACCURACY': False,
#     })
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# # Random swaps max iter sweep at 100, 1k and 10k, with and without sort start

# experiments = []
# for mi in [100, 1000, 10000]:
#     for ss in [False, True]:
#         experiments.append({
#             'ALGORITHM': 'random_swaps',
#             'MAX_ITER': mi, 'SWAP_FRACTION': 1/30, 'SORT_START': ss,
#             'BLOCK_ROWS': 1, 'BLOCK_COLS': 8, 'SPARSITY': 0.5,
#             'SKIP_ACCURACY': True,
#         })


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Defaults for everything
# ---------------------------------------------------------------------------

# Set SKIP_SINGLE_LAYER_ACC for every experiment
DEFAULT_SKIP_SINGLE_LAYER_ACC = True
for exp in experiments:
    exp.setdefault('SKIP_SINGLE_LAYER_ACC', DEFAULT_SKIP_SINGLE_LAYER_ACC)

# Set MODEL_NAME for every experiment
for exp in experiments:
    exp['MODEL_NAME'] = MODEL_NAME

# Set N_CALIB_SAMPLES for every experiment
DEFAULT_N_CALIB_SAMPLES = 128
for exp in experiments:
    exp.setdefault('N_CALIB_SAMPLES', DEFAULT_N_CALIB_SAMPLES)

print(f"Total experiments to run: {len(experiments)}")


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
            
        whole_model_validation_time_v = grab('whole_model_validation_time', '.2f')
        if whole_model_validation_time_v is not None:
            time_str = whole_model_validation_time_v

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
        f"time={time_str}s whole_model_validation_time={whole_model_validation_time_v}s"
    )
