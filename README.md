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
