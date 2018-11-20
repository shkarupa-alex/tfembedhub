# tfembedhub

Convert embeddings vectors (.txt format) into TensorFlow frozen embedding lookup.

## How to

1. Save lookup keys and embeddings values in text file.

Keys should be in first column. Other columns treated as embedding values. Any space-like characters allowed as columns separator.

Important: first row should have key "<UNQ>".

```
<UNQ> 0. 0. 0.
key1 1. 2. 3.
key2 4. 5. 6.
```

2. Use command "tfembedhub-convert" to convert saved array into TF Hub Module.
```bash
tfembedhub-convert vectors.txt vectors-hub/
```
