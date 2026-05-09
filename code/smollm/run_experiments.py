"""Run SmolLM2 pruning experiments via Papermill.

Each experiment runs Smollm-2-pruning-and-eval.ipynb with parameters and collects:
  perplexity, baseline_perplexity, execution_time,
  relative_errors (list per layer), and the path to the per-layer metrics CSV
  the notebook writes (used here to compute mean score_improvement_pct).

Edit the `experiments` list to control what gets run.
"""

import os
import papermill as pm
import scrapbook as sb


# ============================================================================
# Experiment list - choose which experiments to run
# ============================================================================
# algorithm options:
# 'original_tetris', 'our_tetris', 'random_swaps', 'sort_columns_by_norm', 'block_wanda', 'random_swaps'

# ---------------------------------------------------------------------------
# Choose model
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


# ---------------------------------------------------------------------------
# Example experiment
experiments = [
    {
        'ALGORITHM': 'block_wanda',
        'BLOCK_ROWS': 1,
        'BLOCK_COLS': 2,
        'SPARSITY': 0.5,
        'SKIP_PERPLEXITY': True,
        'MODEL_NAME': MODEL_NAME,
    }
]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# # Full block sweep
# experiments = []

# BLOCK_COLS_SWEEP = [2, 4, 8, 16, 32]
# BLOCK_ROWS_FIXED = 1
# SPARSITY_FIXED = 0.5

# alg_configs = [
#     {'ALGORITHM': 'block_wanda'},
#     {'ALGORITHM': 'sort_columns_by_norm'},
#     {'ALGORITHM': 'original_tetris', 'MAX_ITER': 10},
#     {'ALGORITHM': 'our_tetris',
#      'MAX_ITER': 5, 'RANDOM_SWAPS': 10, 'INNER_REFINE': 2},
#     {'ALGORITHM': 'random_swaps',
#      'MAX_ITER': 1000, 'SWAP_FRACTION': 1/30, 'SORT_START': False},
#     {'ALGORITHM': 'random_swaps',
#      'MAX_ITER': 1000, 'SWAP_FRACTION': 1/30, 'SORT_START': True},
# ]

# for block_cols in BLOCK_COLS_SWEEP:
#     for cfg in alg_configs:
#         experiments.append({
#             **cfg,
#             'BLOCK_ROWS': BLOCK_ROWS_FIXED,
#             'BLOCK_COLS': block_cols,
#             'SPARSITY': SPARSITY_FIXED,
#             'SKIP_PERPLEXITY': False,
#         })

# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# # Our tetris, max iter = 5, random swaps = 100, just for 1x2
# experiments = [
#     {
#         'ALGORITHM': 'our_tetris',
#         'MAX_ITER': 5, 'RANDOM_SWAPS': 100, 'INNER_REFINE': 2,
#         'BLOCK_ROWS': 1, 'BLOCK_COLS': 2, 'SPARSITY': 0.5,
#         'SKIP_PERPLEXITY': False,
#     }
# ]

# ---------------------------------------------------------------------------
# # Run 1x2 all algs
# experiments = []
# DEFAULTS = {
#     'BLOCK_ROWS': 1,
#     'BLOCK_COLS': 2,
#     'SPARSITY': 0.5,
#     'SKIP_PERPLEXITY': False,
#     'MODEL_NAME': MODEL_NAME,
# }
# alg_configs = [
#     {'ALGORITHM': 'block_wanda'},
#     {'ALGORITHM': 'sort_columns_by_norm'},
#     {'ALGORITHM': 'original_tetris', 'MAX_ITER': 10},
#     {'ALGORITHM': 'our_tetris', 'MAX_ITER': 5, 'RANDOM_SWAPS': 100, 'INNER_REFINE': 2},
#     {'ALGORITHM': 'random_swaps', 'MAX_ITER': 1000, 'SWAP_FRACTION': 1/30, 'SORT_START': False},
#     {'ALGORITHM': 'random_swaps', 'MAX_ITER': 1000, 'SWAP_FRACTION': 1/30, 'SORT_START': True}, 
# ]

# for cfg in alg_configs:
#     experiments.append({**DEFAULTS, **cfg})

# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------

print(f"Total experiments: {len(experiments)}")


# ============================================================================
# Output setup — folders match what the notebook writes to
# ============================================================================
# Mirror the notebook's _safe_name logic so we know which folder it will use.
def _safe_name(s):
    return s.replace("/", "_").replace(":", "_")

OUTPUT_FOLDER = f"experiment_results_smollm_{_safe_name(MODEL_NAME)}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
summary_file = os.path.join(OUTPUT_FOLDER, "results_summary.md")

HEADER = (
    "| Algorithm | Iters | Swaps | InnerRef | SortStart | Sparsity | Block "
    "| Avg RelErr | ScoreImprov(mean) | Perplexity | Time(s) |\n"
)
DIVIDER = "|" + "|".join(["---"] * 11) + "|\n"

if not os.path.exists(summary_file):
    with open(summary_file, "w") as f:
        f.write(HEADER)
        f.write(DIVIDER)
else:
    with open(summary_file, "a") as f:
        f.write(DIVIDER)


# ============================================================================
# Run loop
# ============================================================================
for exp in experiments:
    alg = exp['ALGORITHM']
    if alg == 'random_swaps' and exp.get('SORT_START', False):
        alg_label = alg + '_sort_start'
    else:
        alg_label = alg

    iters = exp.get('MAX_ITER', 'na')
    swaps = exp.get('RANDOM_SWAPS', 'na')
    inner = exp.get('INNER_REFINE', 'na')
    sortstart = exp.get('SORT_START', 'na')

    r = exp['BLOCK_ROWS']
    c = exp['BLOCK_COLS']
    sparsity = exp['SPARSITY']

    file_name = (
        f"{alg_label}_{r}x{c}"
        f"_iters_{iters}_swaps_{swaps}_innerref_{inner}"
        f"_sortstart_{sortstart}_sparsity_{sparsity}.ipynb"
    )
    output_path = os.path.join(OUTPUT_FOLDER, file_name)

    # # Resume guard: skip if already executed
    # if os.path.exists(output_path):
    #     print(f"Skipping {file_name} (already exists)")
    #     continue

    print(f"Starting {file_name}...")

    perp_str   = "N/A"
    base_str   = "N/A"
    err_str    = "N/A"
    score_str  = "N/A"
    time_str   = "N/A"

    try:
        pm.execute_notebook(
            'Smollm-2-pruning-and-eval.ipynb',
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

        v = grab('perplexity', '.4f');           perp_str  = v if v is not None else "N/A"
        v = grab('baseline_perplexity', '.4f');  base_str  = v if v is not None else "N/A"
        v = grab('execution_time', '.2f');       time_str  = v if v is not None else "N/A"

        try:
            errs = nb.scraps['relative_errors'].data
            err_str = f"{sum(errs)/len(errs):.4f}"
        except KeyError:
            err_str = "N/A"

        # Pull mean score_improvement_pct directly from the CSV the notebook wrote
        try:
            csv_path_v = grab('layer_metrics_csv')
            if csv_path_v:
                import pandas as pd
                df_csv = pd.read_csv(csv_path_v)
                if 'score_improvement_pct' in df_csv.columns:
                    score_str = f"{df_csv['score_improvement_pct'].mean():.3f}"
        except Exception:
            pass

    except Exception as e:
        print(f"Error running {file_name}: {e}")
        perp_str = base_str = err_str = score_str = time_str = "CRASHED"

    with open(summary_file, "a") as f:
        f.write(
            f"| {alg_label} | {iters} | {swaps} | {inner} | {sortstart} "
            f"| {sparsity} | ({r}x{c}) | {err_str} | {score_str} "
            f"| **{perp_str}** (base={base_str}) | {time_str} |\n"
        )

    print(
        f"Finished {file_name}!  "
        f"score={score_str}%  perp={perp_str}  base={base_str}  time={time_str}s"
    )
