# tfembedhub

Convert embeddings vectors (.txt format) into TensorFlowHub embedding lookup module.

## How to

1. Save lookup keys and embeddings values in text file.

Keys should be in first column. Other columns treated as embedding values. Any space-like characters allowed as columns separator.

Non-existing keys will refer to "<UNQ>" key embeddings. You may provide embedding values for that, otherwise it will be initialized with zeros.

```
key1 1. 2. 3.
key2 4. 5. 6.
<UNQ> 0. -1. 0.
```

2. Sonvert saved embeddings into TF Hub Module with "tfembedhub-convert" command.
```bash
tfembedhub-convert vectors.txt vectors-hub/
```

3. Use embedding hub via columns in your estimator.
```python
from tfembedhub text_embedding_column, sequence_text_embedding_column

my_words_embedding = sequence_text_embedding_column(
    key='sparse_key_from_features',
    module_spec='path/to/my/hub'
)

# Then pass my_words_embedding to estimator "columns" list.
```