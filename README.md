# M-Net

M-Net is a PyTorch-based deep learning project for spatiotemporal sequence forecasting. This repository provides model implementations, data loading utilities, training and evaluation pipelines, and visualization tools.

## Repository Structure

```
M-Net/
├── data_provider/
│   ├── data_factory.py
│   └── data_loader.py
│
├── dataset/
│   ├── download_ERA5.txt
│   ├── download_bair.sh
│   ├── download_human3.6m.sh
│   ├── download_kinetics.sh
│   ├── download_kitticaltech.sh
│   ├── download_kth.sh
│   ├── download_mfmnist.sh
│   ├── download_mmnist.sh
│   ├── download_sevir.sh
│   ├── download_taxibj.sh
│   └── download_weatherbench.sh
│
├── model/
│   ├── CNN.py
│   ├── CUNet.py
│   ├── MNet.py
│   └── Transformer.py
│
├── openstl/
│   ├── api/
│   ├── core/
│   ├── datasets/
│   │   └── pipelines/
│   ├── methods/
│   ├── models/
│   ├── modules/
│   │   └── layers/
│   └── utils/
│
├── tool/
│   ├── DPA.py
│   ├── SC-BiGRU.py
│   ├── attention.py
│   ├── map.py
│   ├── metrics.py
│   ├── normalization.py
│   └── plot.py
│
├── main.py
├── run.py
├── environment.yml
└── README.md
```

### Key Components

| Directory / File  | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `data_provider/`  | Data loading and preprocessing pipeline                      |
| `dataset/`        | Dataset download instructions and preparation scripts        |
| `model/`          | Model architecture implementations                           |
| `openstl/`        | OpenSTL framework integration and supporting components      |
| `tool/`           | Attention modules, prediction utilities, metrics, normalization, and visualization |
| `main.py`         | Training, validation, and testing implementation             |
| `run.py`          | Main experiment entry point                                  |
| `environment.yml` | Conda environment and dependency configuration               |

## Installation

### 1. Clone the Repository

```
git clone https://github.com/Dr-BinYang/M-Net.git
cd M-Net
```

### 2. Create the Environment

Create the Conda environment using the provided configuration:

```
conda env create -f environment.yml
```

Activate the environment:

```
conda activate pyg
```

**Environment Requirements**

The provided environment configuration includes:

- Python 3.11.10
- PyTorch 2.6.0
- CUDA 12.6
- PyTorch Geometric 2.6.1

## Running Experiments

### 1. Default Configuration

Run the following command to start the training process:

```
python run.py
```

### 2. Custom Configuration

Experimental parameters can be modified through command-line arguments defined in `run.py`.

For example:

```
python run.py \
    --model MNet \
    --seq_len 6 \
    --pred_len 5 \
    --batch_size 64 \
    --learning_rate 0.001
```



