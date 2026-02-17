## HuggingFaceTB/SmolLM2-135M

- source: https://huggingface.co/HuggingFaceTB/SmolLM2-135M

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
