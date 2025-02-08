import numpy as np

data = {np.int64(10): 43.7709145461421, np.int64(191): 27.932993776271914}

# Get the key with the minimum value
min_key = min(data, key=data.get)

print(min_key)  # Output: 191