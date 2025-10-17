# M-Net

M-Net: A deep learning framework for spatiotemporal sequence forecasting.

## 📁 Project Structure  

M-Net/
├── data_provider/          # Data loading and preprocessing utilities
│   └── ...                 # Data pipeline components
├── dataset/                # Dataset definitions and management
│   └── ...                 # Dataset-specific implementations
├── model/                  # Core model architecture
│   └── ...                 # M-Net implementation modules
├── openstl/                # OpenSTL framework integration
│   └── ...                 # Base framework components
├── tool/                   # Utility scripts
│   └── ...                 # Helper functions and tools
├── main.py                 # Main training script
├── run.py                  # Inference/execution script
├── environment.yml         # Conda environment configuration
└── README.md               # Project documentation

## 🧰 Key Components  
- **`data_provider/`**: Data loading and preprocessing module  
- **`dataset/`**: Dataset definition and management system  
- **`model/`**: Core M-Net architecture implementation  
- **`openstl/`**: Integrated OpenSTL framework components  
- **`tool/`**: Utility function collection  

## 🚀 Quick Start  
```bash
# Create environment
conda env create -f environment.yml
conda activate mnet

# Start training
python main.py

# Run inference
python run.py
