## NEW: Running on very small model

model_name = "test_vit3.r160_in1k"
original (before pruning) accuracy = 0.56922

### One layer pruned

#### Settings

- BLOCK_SIZE = (1, 8)
- SPARSITY = 0.5
- MAX_ITER = 100
- RANDOM_SWAPS = 5000 (50k for "Block + random swaps")
- if there is permutation, it is permutation of columns

#### Comprehensive comparative table

| Method                               | Sum of Pruned Weights (Lower is Better) | Improvement vs Standard\* | Final Val Accuracy | Acc Drop vs Original |
| :----------------------------------- | :-------------------------------------- | :------------------------ | :----------------- | :------------------- |
| **Our algorithm with swaps**         | **353.04**                              | **19.82%**                | 42.13%             | -14.79%              |
| **Block pruning + random swaps**     | 375.94                                  | 14.62%                    | **54.14%**         | **-2.78%**           |
| **Original tetris (our impl)**       | 387.59                                  | 11.97%                    | 46.80%             | -10.12%              |
| **Sort columns by weight**           | 412.61                                  | 6.29%                     | 53.85%             | -3.07%               |
| _Standard Block Pruning (Reference)_ | _440.30_                                | _0.0%_                    | _N/A_              | _N/A_                |

> **\*Note:** "Improvement vs Standard" means we saved X% more weight magnitude from being pruned compared to the standard block pruning.\*

#### Some notes from results

- First point of results for a specific alg is copy of the last line of output of that alg where pruned sum improved (lowered) with headers
- _Total diff %_ is the diff from first iteration of block pruning

##### Our alg with swaps:

- alg: tetris + noise + random swaps iterated with finding optimal permutation
- AFTER_SWAP, DIFF, DIFF %, TOTAL DIFF AFTER TETRIS %, TOTAL DIFF % -> 353.0371704102, 0.0000000000, 0.0000000000, 4.5957064629, **19.8182334900**
- val acc 0.4213
- cca 1000 random swaps iterations

##### Block pruning + random swaps:

- alg: start with identity permutation, iterate calculating mask and doing $x=10$ random swaps in the permutation
- AFTER_SWAP, DIFF, DIFF %, TOTAL DIFF % -> 375.9367065430, 1.8645629883, 0.4935300946, **14.6172943115**
- val acc 0.5414
- (cca 20k iterations)

##### Original tetris (our impl):

- BLOCK, ORIG TETRIS, DIFF, DIFF %, TOTAL DIFF % -> 387.5853271484, 387.5853271484, 0.0000000000, 0.0000000000, **11.9716606140**
- val acc 0.46804

##### Sort columns by weight

- alg: sort (descending) columns by sum of absolute values of their weights (L1 norm)
- ORIGINAL, AFTER SORT, DIFF, DIFF % -> 440.2960815430, 412.6115722656, 27.6845092773, **6.2877030373**
- val acc 0.53854
- 1 iteration (doesnt make sense to do more)

## OLD

### Running Different Algorithms

Change what is called at the end of `tetris.py`

#### Understanding Output

- Without noise, the headings make sense
- With noise:
  - After tetris, diff can be negative and still the total diff will be higher
  - This means that after running the tetris algorithm, the pruned value increased compared to what it was after finding the mask in this iteration
  - That is okay because it seems that after finding a new mask in the next iteration, the overall pruned sum will often improve
  - Just the headings are a bit confusing with this
- Random swaps after tetris:
  - There are new headings that make sense
- Just random swaps:
  - The headings make sense

#### Adding Noise

- We add noise to the weight matrix before calculating gains and finding the best permutation
- Then we use this permutation on the original weights
- This achieves that we don't find the best permutation, but some $k$-th best

#### Noise Adding Algorithm Modifications

- We can choose between normal and uniform distribution for the noise
- Also, we can change the function by which the noise percentage decreases over iterations

#### Some Results With Noise

Tried different init percentages; so far, $25\%$ looks the best

##### 15% Init Noise

Only ran it 1-2 times, not that reliable results.
Ran on $[2,2000]$ shape, $5000$ iterations, $(1,32)$ blocks
(Normal dist diff / Uniform dist diff)

- Linear: `initial_noise - progress` ~4.8%, ~3.8% diff
- Cosine: `initial_noise * np.cos(progress * np.pi/2)` -> ~4.6% diff, ~3.5% diff
- Inv_sqrt: `initial_noise / np.sqrt(1 + 10 * progress)` -> ~4.2%$ diff, ~3.4% diff

##### 25% Init Noise

- Linear and normal: ~5.7%
