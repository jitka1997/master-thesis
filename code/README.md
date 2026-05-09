# Code used in thesis

Code for the master's thesis _Permuting Matrix Columns for Improved Block-Sparse Pruning_. Implements GA-TETRIS and baselines (Original TETRIS, Sort-by-Norm, Random-Swaps, Block-Wanda) and evaluates them on two ViTs and SmolLM2-360M.

## Structure

```
code/
├── tetris.py              # algorithm implementations
├── vit/
│   ├── ViT-pruning-and-eval.ipynb
│   └── run_vit_experiments.py
├── smollm/
│   ├── Smollm-2-pruning-and-eval.ipynb
│   ├── run_experiments.py
│   ├── modelutils.py
│   └── datautils.py
└── csvs_for_thesis/
    ├── cross_algorithm_analysis.ipynb
    ├── hyperparameters_sensitivity.ipynb
    └── benchmark_csvs_*/
```

Algorithm names accepted by `ALGORITHM`: `our_tetris` (GA-TETRIS), `original_tetris`, `sort_columns_by_norm`, `random_swaps`, `block_wanda`.

## Requirements

Python 3.11, PyTorch 2.x. Other packages: `timm`, `transformers`, `datasets`, `scipy`, `papermill`, `scrapbook`, `pandas`, `numpy`, `matplotlib`, `tqdm`, `jupyter`.

ViT requires ImageNet — set `IMAGENET_PATH` in `ViT-pruning-and-eval.ipynb`. Download from [HuggingFace](https://huggingface.co/datasets/ILSVRC/imagenet-1k) or [ImageNet2012-download](https://github.com/DoranLyong/ImageNet2012-download). SmolLM2 downloads C4 / WikiText-2 automatically.

## Running

Edit the `experiments` list in the relevant driver, then:

```
cd vit
python run_vit_experiments.py
```

or

```
cd smollm
python run_experiments.py
```

Each run writes per-layer CSVs to `benchmark_csvs_{model}/` and a row to `results_summary.md`.

## Figures

The CSVs from the thesis runs are committed under `csvs_for_thesis/benchmark_csvs_*/`. Run `csvs_for_thesis/cross_algorithm_analysis.ipynb` and `csvs_for_thesis/hyperparameters_sensitivity.ipynb` to regenerate the thesis figures from them. Both notebooks save PDFs to `../../thesis/images/` — adjust `FIGURES_DIR` if needed.

New experiment runs write their CSVs to `benchmark_csvs_{model}/`, separate from the committed thesis CSVs.
