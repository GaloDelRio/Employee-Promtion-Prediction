from __future__ import annotations

import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='Custom')
class SliceColumn(layers.Layer):
    def __init__(self, index: int, keepdims: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.keepdims = keepdims

    def call(self, inputs):
        if self.keepdims:
            return inputs[:, self.index:self.index + 1]
        return inputs[:, self.index]

    def get_config(self):
        config = super().get_config()
        config.update({'index': self.index, 'keepdims': self.keepdims})
        return config


def transformer_block(x, emb_dim: int, num_heads: int, ff_dim: int, dropout: float, name: str):
    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=max(1, emb_dim // num_heads),
        dropout=dropout,
        name=f'{name}_mha',
    )(x, x)

    x = layers.Add(name=f'{name}_add_1')([x, attention])
    x = layers.LayerNormalization(name=f'{name}_ln_1')(x)

    ff = layers.Dense(ff_dim, activation='relu', name=f'{name}_ff_1')(x)
    ff = layers.Dropout(dropout, name=f'{name}_dropout')(ff)
    ff = layers.Dense(emb_dim, name=f'{name}_ff_2')(ff)

    x = layers.Add(name=f'{name}_add_2')([x, ff])
    x = layers.LayerNormalization(name=f'{name}_ln_2')(x)
    return x


def build_saint_like_model(
    cat_dims,
    num_numeric: int,
    emb_dim: int = 8,
    num_heads: int = 1,
    num_layers: int = 1,
    dropout: float = 0.1,
) -> keras.Model:
    if emb_dim % num_heads != 0:
        raise ValueError('emb_dim debe ser divisible entre num_heads.')

    total_cat = len(cat_dims)

    cat_input = keras.Input(shape=(total_cat,), dtype='int32', name='categorical_inputs')
    num_input = keras.Input(shape=(num_numeric,), dtype='float32', name='numeric_inputs')

    tokens = []

    for i, dim in enumerate(cat_dims):
        col = SliceColumn(index=i, keepdims=False, name=f'cat_slice_{i}')(cat_input)
        emb = layers.Embedding(input_dim=dim, output_dim=emb_dim, name=f'cat_emb_{i}')(col)
        emb = layers.Reshape((1, emb_dim), name=f'cat_reshape_{i}')(emb)
        tokens.append(emb)

    for i in range(num_numeric):
        col = SliceColumn(index=i, keepdims=True, name=f'num_slice_{i}')(num_input)
        proj = layers.Dense(emb_dim, name=f'num_proj_{i}')(col)
        proj = layers.Reshape((1, emb_dim), name=f'num_reshape_{i}')(proj)
        tokens.append(proj)

    if not tokens:
        raise ValueError('El modelo necesita al menos una columna de entrada.')

    if len(tokens) == 1:
        x = tokens[0]
    else:
        x = layers.Concatenate(axis=1, name='token_concat')(tokens)

    for block_idx in range(num_layers):
        x = transformer_block(
            x,
            emb_dim=emb_dim,
            num_heads=num_heads,
            ff_dim=emb_dim * 2,
            dropout=dropout,
            name=f'transformer_{block_idx + 1}',
        )

    x = layers.Flatten(name='flatten_tokens')(x)
    x = layers.Dense(64, activation='relu', name='classifier_dense')(x)
    x = layers.Dropout(dropout, name='classifier_dropout')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    return keras.Model(
        inputs={'categorical_inputs': cat_input, 'numeric_inputs': num_input},
        outputs=output,
        name='saint_like_classifier',
    )


def get_model_summary_text(model: keras.Model) -> str:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + '\n'))
    return buffer.getvalue()