# Seismic-phase-Classification
This is the implementation for Seismic-phase-Classification

## Data
The dataset including P-waves, S-waves, and pre-event noise classes can be accessed at the Southern California Earthquake Data Center (http://scedc.caltech.edu/research-tools/deeplearning.html) labeled as "**scsn_ps_2000_2017_shuf.hdf5**".

## Model training and testing
- For 1D ResNet model, before running the code, please manually define batch_size, epochs, lr(learnin rate) and the size of training dataset and testing dataset
  ```
  python Resnet_main.py
  ```
- For 1D Multi-branch ResNet, before running the code, please manually define batch_size, epochs, lr(learnin rate) and the size of training dataset and testing 
  ```
  python mb_resnet_main.py
  ```
## Requirements
The models provided here are implemented in Pytorch.
- Pytorch 1.9
- CUDA 11.2 for GPU

## Cite
If you use this code for your research, please cite it with the following bibtex entry.
