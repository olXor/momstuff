#!/bin/bash

echo 'CHILDDEPTH 1' > genbot/gen.cfg
echo 'ALLOW_SIDE_WEIGHTS 0' >> genbot/gen.cfg
echo 'ALLOW_SIDE_MEMS 0' >> genbot/gen.cfg
echo 'MAX_NODESPERLAYER 200' >> genbot/gen.cfg
echo 'MAX_LAYERS 2' >> genbot/gen.cfg
echo 'MAX_NUMPERTURBS 0' >> genbot/gen.cfg
echo 'MAX_CONVOLUTION_LEVELS 2' >> genbot/gen.cfg
echo 'MIN_CONVOLUTION_LEVELS 2' >> genbot/gen.cfg
echo 'MAX_CONVOLUTIONS 3' >> genbot/gen.cfg
echo 'MIN_CONVOLUTIONS 1' >> genbot/gen.cfg
echo 'MAX_CONVOLUTION_NODE_LAYERS 1' >> genbot/gen.cfg
echo 'MAX_CONVOLUTION_NODESPERLAYER 10' >> genbot/gen.cfg
echo 'MAX_CONVOLUTION_DIMENSION 10' >> genbot/gen.cfg
echo 'MIN_CONVOLUTION_DIMENSION 2' >> genbot/gen.cfg
echo 'CONVOLUTION_DIMENSION_LAYER_MULTIPLIER 2.0' >> genbot/gen.cfg
echo 'NUM_TURNS_SAVED 15' >> genbot/gen.cfg
