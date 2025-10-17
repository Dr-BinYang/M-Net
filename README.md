# M-Net

M-Net: A deep learning framework for spatiotemporal sequence forecasting.

## 📁 Project Structure  

M-Net/

  ├── data_provider/          # Data loading and preprocessing utilities
    
  ├── dataset/                # Dataset or download link 
  
  ├── model/                  # models 
  
  │   └── ...                 # M-Net\CUNet\CNN implementation 
  
  ├── openstl/                # OpenSTL framework integration
  
  │   └── ...                 # Base framework components
  
  ├── tool/                   # Utility scripts
  
  │   └── ...                 # Helper functions and tools
  
  ├── main.py                 # Main  script
  
  ├── run.py                  # Run script
  
  ├── environment.yml         # Conda environment configuration
  
  └── README.md               # Project documentation


## 🧰 Key Components  

- **data_provider/**: Complete data preprocessing and loading pipeline
- **dataset/**:  Download links for experimental datasets 
- **model/**: Include M-Net architecture implementation
- **openstl/**: OpenSTL framework 
- **tool/**: Development and maintenance utilities
- **main.py**: Main  script：Include Training、Validation、Test function 
- **run.py**: Run script
- **environment.yml**: Complete development environment configuration

*Note: This structure represents the core framework organization. Additional implementation details can be found in each module's annotation.*

## 🚀 Quick Start  
```bash
# Create environment
conda env create -f environment.yml
conda activate mnet

# Start run
python run.py
