# tfstatlookup

Convert NymPy 2D array of keys and features into TensorFlow frozen embedding lookup.

## How to

1. Save array with features (or statistics) and lookup key as first column.
Important: first row should have key "<-UNIQUE->"

```
import numpy as np

source = np.array([
    ('<-UNIQUE->', 0., 0., 0.),
    ('key1', 1., 2., 3.),
    ('key2', 4., 5., 6.),
])
np.save('source.npy', source)

```

2. Use command "tfstatlookup-convert" to convert saved array into TF Hub Module.
```bash
tfstatlookup-convert source.npy module-dir/
```
