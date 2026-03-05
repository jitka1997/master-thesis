## HuggingFaceTB/SmolLM2-135M

- source: https://huggingface.co/HuggingFaceTB/SmolLM2-135M

### Pruning whole model

#### Perplexity

- without pruning: 13.619280
- raw wanda: 32.314201
- block wanda, global sparsity per matrix: 204097.218750
- block wanda, sparsity per row: 59270.402344
  - (1, 4): 22101.017578
  - (1, 2): 304.266998
- block wanda tetris: 16998730
- block wanda tetris rowwise sparsity: 2598046.5
  - fix mask and permutation: 669209.062500
  - no noise: 48232.285156
  - (1,4): 23132.328125
  - (1,4), 100 iter, 100 random swaps: 12319.26
  - (1,2): 100 iter, 100 random swaps: 315.520416
    - with noise: 309.973389 (0.5 \* (1 - progress), min 0)
    - with noise: 240.593323 (1 - progress, min 0,1)

### Running on one matrix

#### Settings

- BLOCK_SIZE_BENCH = (1, 8)
- SPARSITY_BENCH = 0.5
- MAX_ITER_BENCH = 100
- RANDOM_SWAPS_BENCH = 5000

#### Graph comparison - (576, 576)

![Comparison Graph](<smollm-(576,576).png>)

#### Graph comparison - (576, 576) - running just random swaps alg 200k iterations

![Comparison Graph - lot of random swaps](<smollm-(576,576)-random_swaps_running_200k_iter.png>)

#### Graph comparison - (192, 576)

![Comparison Graph](<smollm-(192,576)-matrix3.png>)
