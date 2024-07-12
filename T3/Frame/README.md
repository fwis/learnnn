## Introduction
We are trying to build a ResNet-based network for rebuilding I, AoP and DoLP from input DoFP polarization images.
This work is still in an exploration stage.
We are building an end-to-end residual GAN for DoFP polarization image reconstruction.
The code is based on Pytorch.

## Notes
1. You can use 'generate_labels' to process your dataset.
2. The 'test' is for generating full resolution images from a real DoFP image.

## TODO
1. Employ CAM into our network.(Done)
2. Create polarized remote sensing dataset.
3. Apply our network on remote sensing.

# Data
To fit the physics law, DoLP and S0 was normalized in range (0,1), and AoP was normalized in range (0,pi/2).

## Citation

