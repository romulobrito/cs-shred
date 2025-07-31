# CS-SHRED: Compressed Sensing Enhanced SHRED

CS-SHRED (Compressed Sensing - Shallow Recurrent Decoder) is a deep learning framework for spatiotemporal data reconstruction from incomplete measurements, integrating compressed sensing with recurrent neural networks.

## Repository Structure

```
cs-shred/
├── models.py              # CS-SHRED and SHRED model architectures
├── processdata.py         # Data loading and preprocessing utilities
├── turb_flow_csshred.py   # Main training script for turbulence data
├── requirements.txt       # Python dependencies
├── figs/                  # Architecture diagrams and figures
└── results/               # Output directory (created during execution)
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place your spatiotemporal data in `.npy` format
   - Ensure data shape is `(time, height, width)`

3. **Run training:**
   ```bash
   python turb_flow_csshred.py
   ```

## Key Features

- **CS-SHRED Model**: Enhanced SHRED with compressed sensing integration
- **SHRED Model**: Baseline shallow recurrent decoder
- **Flexible Architecture**: Configurable LSTM layers and hidden dimensions
- **Robust Training**: Early stopping and validation monitoring
- **Comprehensive Evaluation**: Multiple metrics (SSIM, PSNR, LPIPS, Normalized Error)

## Model Architecture

![CS-SHRED Architecture](figs/arch-design.png)

The CS-SHRED model integrates:
- Compressed sensing for sparse representation
- LSTM networks for temporal dynamics
- Dense layers for spatial reconstruction

## Data Requirements

- **Format**: NumPy arrays (`.npy` files)
- **Shape**: `(time_steps, spatial_dim_x, spatial_dim_y)`
- **Type**: Float32 recommended
- **Preprocessing**: Min-Max scaling applied automatically

## Configuration

Key hyperparameters can be adjusted in the training script:

- `hidden_size`: LSTM hidden dimensions (default: 64)
- `hidden_layers`: Number of LSTM layers (default: 2)
- `num_epochs`: Training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `lr`: Learning rate (default: 0.001)
- `lags`: Temporal window size (default: 20)
- `num_sensors`: Number of sensors (default: 200)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{daSilva2025csshred,
  title   = {{CS-SHRED}: Enhancing SHRED for Robust Recovery of Spatiotemporal Dynamics},
  author  = {da Silva, R. Brito and Passos, D. and Oishi, C. M. and Kutz, J. N.},
  journal = {arXiv preprint},
  year    = {2025},
  eprint  = {2507.22303},
}
```

## License

This project is open source. Please check the license file for details.

## Contact

For questions or support:
- Romulo B. da Silva: romulo.silva@peq.coppe.ufrj.br
- Repository Issues: [GitHub Issues](https://github.com/romulobrito/cs-shred/issues) 