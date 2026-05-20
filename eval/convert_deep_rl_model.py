#!/usr/bin/env python3
"""
Convert chess_deep_rl model from channels_first (NCHW) to channels_last (NHWC).
NCHW is not supported on CPU (including Apple Silicon), so this one-time conversion
produces model_nhwc.h5 which uses NHWC throughout.

The model interprets its (8, 8, 19) input as (C=8, H=8, W=19) in channels_first mode.
The equivalent channels_last layout is (H=8, W=19, C=8), so callers must transpose
the state array from (batch, 8, 8, 19) to (batch, 8, 19, 8) before calling predict().
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

import numpy as np
import tf_keras as keras
from tf_keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Input, add
from tf_keras.models import Sequential, Model


N_FILTERS = 256
N_RESIDUAL = 19
INPUT_SHAPE_NHWC = (8, 19, 8)   # same data as NCHW (8,8,19) but in H,W,C order
HEAD_INPUT_NHWC = (8, 19, N_FILTERS)
POLICY_OUTPUT = 4672
VALUE_OUTPUT = 1


def _conv_bn_relu(x, filters, kernel_size=(3, 3)):
    x = Conv2D(filters, kernel_size, padding='same', data_format='channels_last', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x


def _residual_block(inp):
    x = Conv2D(N_FILTERS, (3, 3), padding='same', data_format='channels_last', use_bias=False)(inp)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(N_FILTERS, (3, 3), padding='same', data_format='channels_last', use_bias=False)(x)
    x = BatchNormalization(axis=-1)(x)
    x = add([x, inp])
    x = Activation('relu')(x)
    return x


def build_nhwc_model():
    inp = Input(shape=INPUT_SHAPE_NHWC, name='main_input')
    x = _conv_bn_relu(inp, N_FILTERS)
    for _ in range(N_RESIDUAL):
        x = _residual_block(x)

    policy_head = Sequential(name='policy_head')
    policy_head.add(Conv2D(2, (1, 1), padding='same', data_format='channels_last',
                           input_shape=HEAD_INPUT_NHWC, use_bias=True))
    policy_head.add(BatchNormalization(axis=-1))
    policy_head.add(Activation('relu'))
    policy_head.add(Flatten())
    policy_head.add(Dense(POLICY_OUTPUT, activation='sigmoid', name='policy_head'))

    value_head = Sequential(name='value_head')
    value_head.add(Conv2D(1, (1, 1), padding='same', data_format='channels_last',
                          input_shape=HEAD_INPUT_NHWC, use_bias=True))
    value_head.add(BatchNormalization(axis=-1))
    value_head.add(Activation('relu'))
    value_head.add(Flatten())
    value_head.add(Dense(256))
    value_head.add(Activation('relu'))
    value_head.add(Dense(VALUE_OUTPUT, activation='tanh', name='value_head'))

    return Model(inputs=inp, outputs=[policy_head(x), value_head(x)])


def convert(src_path: str, dst_path: str):
    print(f"Loading NCHW model from {src_path} ...")
    old_model = keras.models.load_model(src_path, compile=False)

    print("Building NHWC model ...")
    new_model = build_nhwc_model()

    old_weights = old_model.get_weights()
    new_weights = new_model.get_weights()

    if len(old_weights) != len(new_weights):
        raise ValueError(
            f"Weight count mismatch: old={len(old_weights)}, new={len(new_weights)}. "
            "Architecture may have changed."
        )

    for i, (ow, nw) in enumerate(zip(old_weights, new_weights)):
        if ow.shape != nw.shape:
            raise ValueError(f"Weight {i} shape mismatch: old={ow.shape}, new={nw.shape}")

    new_model.set_weights(old_weights)
    new_model.save(dst_path)
    print(f"Saved NHWC model to {dst_path}")


if __name__ == "__main__":
    src = os.path.join(project_root, "models", "chess_deep_rl", "model.h5")
    dst = os.path.join(project_root, "models", "chess_deep_rl", "model_nhwc.h5")
    convert(src, dst)