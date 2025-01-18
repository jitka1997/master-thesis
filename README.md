### Running diffrent algorithms

Change what is called at the end of `tetris.py`

#### Understading output

- Without noise the headings make sense
- With noise
  - After tetris diff can be negative and still the total diff will be higher
  - This means that after running the tetris alg, the pruned value got higher compared to what it was after fiding mask in this iteration
  - That is okay, because it seems that after finding new mask in the next iteration the overall pruned sum will offen improve
  - Just the headings are a bit confusing with this
- Random swaps after tetris
  - There are new headings that make sense
- Just random swaps
  - The headings make sense

#### Adding noise

- We add noise to weight matrix before calculating ganes and finding the best permutation
- Then we use this permutation on the original weights
- This achieves that we dont find the best permutation, but some $k$-th best

#### Noise adding algorithm modifications

- We can choose between normal and uniform distribution for the noise
- Also we can change the function by with the noise percentage is decreasing over iterations

#### Some results

Tried also diffrent init percentages, so far $25\%$ looks the best

##### 15% init noise

Only ran it 1-2 times, not that relialbe results
Ran on $[2,2000]$ shape, $5000$ iterations, $(1,32)$ blocks
(Normal dist diff / Uniform dist diff)

- Linear: `initial_noise - progress` $~4.8\%$, $~3.8\%$ diff
- Cosine: `initial_noise * np.cos(progress * np.pi/2)` -> $~4.6\%$ diff, $~3.5\%$ diff
- Inv_sqrt: `initial_noise / np.sqrt(1 + 10 * progress)` -> $~4.20\%$ diff, $~3.4\%$ diff

##### 25% init noise

- Linear and normal: $~5.7\%$
