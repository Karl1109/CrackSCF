# Staircase Cascaded Fusion of Lightweight Local Pattern Recognition and Long-Range Dependencies for Structural Crack Segmentation

## Requirements

### Environment requirements: 

- CUDA 11.8

- Python 3.8

### Dependency requirements: 

- numpy 1.24.1
- thop 0.1.1
- tqdm 4.66.1
- opencv-python 4.8.1.78
- einops 0.7.0
- torchaudio 2.1.1
- torchinfo 1.8.0
- torchsummary 1.5.1
- torchvision 0.16.1
- cython 3.0.6
- scipy 1.10.1

## Installation

We recommend you to use Anaconda to create a conda environment:

```
conda create -n crackscf python=3.8 pip
```

Then, activate the environment:

```
conda activate crackscf
```

Install the torch, torchvision and torchaudio

``````
pip install torch==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
``````

Now, you can install other requirements:

``````
pip install -r requirements.txt
``````

# Compiling the CUDA operators of deformable attention

``````
cd ./models/ops
sh ./make.sh
``````

# Train

``````
python train.py
``````

# Test

``````
python test.py
``````

## TUT dataset

The TUT dataset is available at [TUT](https://github.com/Karl1109/TUT)

## Contact

Any questions, please contact the email at liuhui1109@stud.tjut.edu.cn
