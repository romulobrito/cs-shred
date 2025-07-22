# CS-SHRED: TURB-ROT

This repository provides a minimal example to run the CS-SHRED pipeline on the TURB-ROT dataset.

## How to Run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the TURB-ROT dataset:
   - The file `data/turb_vy_combined.npy` is too large to be included in the repository.
   - Download it from: [Google Drive – turb_vy_combined.npy](https://drive.google.com/file/d/1CKhvfhdQpOFiXHGfu-8FOTPzSgghQmlB/view?usp=drive_link)
   - Place the downloaded file in:
     ```
     data/turb_vy_combined.npy
     ```


3. Run the pipeline:
   ```bash
   python turb_flow_csshred.py
   ```

## Main Files
- `turb_flow_csshred.py`: Main script for the TURB-ROT CS-SHRED pipeline
- `models.py`: Model architectures
- `processdata.py`: Utility functions for data loading and preparation
- `requirements.txt`: Python dependencies

## Notes
- The dataset is not distributed due to its size. Please use the link above to download it.
- Results and logs will be saved in the `results/` folder. 
