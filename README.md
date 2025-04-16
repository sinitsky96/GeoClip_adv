# GeoClip_adv

## Overview

This project implements adversarial attacks on geolocation models using modified versions of SparseRS and PGDTrimKernel approaches. The implementation focuses on sparse patch-based attacks with various configurations and optimization strategies.

## Environment Setup

- Python Version: 3.11.11
- Environment: Mamba
- Dependencies: See `requirements.txt`

## Project Structure

The project consists of several key components:
- SparsePatches implementation for adversarial attacks
- Modified SparseRS framework adapted for geolocation
- Various attack configurations and parameters

## Attack Implementation

### SparsePatches Attack

The implementation supports flexible attack configurations:

#### Supported Parameters
- **Kernel Sizes**: 1x1, 2x2, and 4x4
- **Sparsity Levels**: 16, 64, 128, and 256
- **Batch Processing**: Configurable batch sizes
- **Optimization**: Various trimming steps and mask distributions

#### Key Command Line Arguments
- `--att_kernel_size`: Attack kernel size (1, 2, or 4)
- `--sparsity`: Number of patches (16, 64, 128, or 256)
- `--att_trim_steps`: Optimization trimming steps sequence
- `--eps_l_inf_from_255`: L∞ constraint epsilon value (default: 8)

### SparseRS Framework

Our implementation builds upon the SparseRS framework, a versatile approach for query-efficient sparse black-box adversarial attacks.

#### Attack Types

1. **L0-bounded Attacks**
   - Pixel/feature-level modifications
   - Configurable number of modifications via `k` parameter
   - Supports targeted and untargeted attacks

2. **Patch Attacks**
   - Image-specific and universal patch support
   - Flexible patch sizes (20x20, 40x40, 50x50)
   - Both targeted and untargeted versions

3. **Frame Attacks**
   - Adversarial frame generation
   - Adjustable frame width
   - Image-specific and universal options

#### Framework Modifications

We've adapted the original SparseRS framework with the following changes:

1. **Domain Adaptation**
   - Optimized for geolocation attacks
   - Enhanced patch placement support
   - Multiple simultaneous patch modifications

2. **Parameter Relationships**
   - Sparse attacks (1x1): k represents epsilon
   - Patch attacks: epsilon remains constant
   - Kernel area calculation: $kernel area = \sqrt(k)\times \sqrt(k) < \epsilon$ ($L_0 limit$)

## Running the Code

### Basic Usage

Run examples are provided in the `run_*.sh` files. Default settings from the original papers are maintained where applicable.

### Data Preparation

1. **General Datasets**
   - Use provided download.py scripts in dataset directories
   - Follow standard data loading procedures

2. **Im2GPS3k Dataset**
   - Manual download required
   - Source: https://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip
   - Extract to: `/data/Im2GPS3k/images`

## Technical Details

### Implementation Notes

The project incorporates modifications to both SparseRS and PGDTrimKernel:
- Adapted for geolocation domain requirements
- Enhanced patch placement capabilities
- Modified kernel area and epsilon relationships

### Mathematical Framework

For patch-based attacks:
- Kernel Area Constraint: $kernel area = \sqrt(k)\times \sqrt(k) < \epsilon$
- Sparse Attack Case: k represents epsilon
- Patch Attack Case: k represents kernel area size in pixels