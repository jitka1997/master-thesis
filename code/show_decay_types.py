import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def get_noise_percentage(initial_noise, current_iter, max_iter, decay_type='exponential'):
    progress = current_iter / max_iter
    
    if decay_type == 'exponential':
        return initial_noise * np.exp(-5 * progress)
    
    elif decay_type == 'cosine':
        return initial_noise * np.cos(progress * np.pi/2)
    
    elif decay_type == 'inv_sqrt':
        return initial_noise / np.sqrt(1 + 10 * progress)
    
    else:
        raise ValueError("decay_type must be 'exponential', 'cosine', or 'inv_sqrt'")
    
iters = np.linspace(0, 5000, 5001)  # 101 points to include both 0 and 100
exp_decay = [get_noise_percentage(15, i, 5000, 'exponential') for i in iters]
cos_decay = [get_noise_percentage(15, i, 5000, 'cosine') for i in iters]
sqrt_decay = [get_noise_percentage(15, i, 5000, 'inv_sqrt') for i in iters]
linear = [15 - 15 * i/5000 for i in iters]

plt.plot(exp_decay, label='Exponential')
plt.plot(cos_decay, label='Cosine')
plt.plot(sqrt_decay, label='Inverse sqrt')
plt.plot(linear, label='Linear')
plt.xlabel('Iteration')
plt.ylabel('Noise percentage')
plt.legend()
plt.show()
