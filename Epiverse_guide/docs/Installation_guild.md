# Installation

## Prerequisites


EpiVerse can be installed via conda or pypi and you need to install `pytorch` at first

!!! note 
    To avoid potential dependency conflicts, installing within a `conda` environment is recommended. And using `pip install -U epiverse` to update.

### Platform

In different platform, there are some differences in the most appropriate installation method.

- `Windows`: We recommend installing the [`wsl` subsystem](https://learn.microsoft.com/en-us/windows/wsl/install) and installing `conda` in the wsl subsystem to configure the omicverse environment.
- `Linux`: We can choose to install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html), and then use conda to configure the omicverse environment
- `Mac Os`: We recommend using [`miniforge`](https://github.com/conda-forge/miniforge)  or [`mambaforge`](https://www.rho-signal-effective-analytics.com/modules/pre-course/miniconda-installation/)to configure.

### pip prerequisites
- If using conda/mamba, then just run `conda install -c anaconda pip` and skip this section.
- Install Python, we prefer the pyenv version management system, along with pyenv-virtualenv.

### Apple silicon prerequisites
Installing omicverse on a Mac with Apple Silicon is only possible using a native version of python. A native version of python can be installed with an Apple Silicon version of mambaforge (which can be installed from a native version of homebrew via `brew install --cask mambaforge`). 

### Dev package

You need to install `python3.8-dev` on linux enviroment 

```shell
sudo apt-get install python3.8-dev
```


## Conda

Under development

## Pip

The `epiverse` package can be installed via pip using one of the following commands:

1. Install [PyTorch](https://pytorch.org/get-started/locally/) at first: More about the installation can be found at [PyTorch](https://pytorch.org/get-started/locally/). 

   ```shell
   # ROCM 5.2 (Linux only)
   pip3 install torch torchvision torchaudio --extra-index-url
   pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
   # CUDA 11.6
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   # CUDA 11.7
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   # CPU only
   pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
   ```
2. After the installation of pytorch, we can start to install `epiverse` by `pip`

   ```shell
   pip install -U epiverse
   pip install -U numba
   ```
3. If you want to using Nightly verseion. There are two ways for you to install

   - Nightly version - clone this [repo](https://github.com/DBinary/Epiverse) and run: `pip install .`
   - Using `pip install git+https://github.com/DBinary/Epiverse`

## Others

if you using M1/M2 silicon, perhaps the following code will be helped:

```shell
conda install s_gd2 -c conda-forge
pip install -U epiverse 
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

## Development

For development - clone this repo and run:

```shell
pip install -e ".[dev,docs]"
```

