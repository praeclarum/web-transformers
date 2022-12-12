#!/usr/bin/env bash

MODEL_ID=t5-small
MODEL_DIR=public/models

mkdir -p $MODEL_DIR

curl https://transformers-js.praeclarum.org/models/$MODEL_ID-encoder-quantized.onnx --output $MODEL_DIR/$MODEL_ID-encoder-quantized.onnx
curl https://transformers-js.praeclarum.org/models/$MODEL_ID-init-decoder-quantized.onnx --output $MODEL_DIR/$MODEL_ID-init-decoder-quantized.onnx
curl https://transformers-js.praeclarum.org/models/$MODEL_ID-decoder-quantized.onnx --output $MODEL_DIR/$MODEL_ID-decoder-quantized.onnx
