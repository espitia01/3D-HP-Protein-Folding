# 3D-HP-DRL

This project implements a 3D Hydrophobic-Polar (HP) model using Reservoir Deep Reinforcement Learning.

## Getting Started

### Prerequisites

- Anaconda or Miniconda
- CUDA-capable GPU (recommended)

### Installation

1. Create a new conda environment:

conda create -n myenv python=3.8.10

2. Activate the environment:

conda activate myenv

3. Install dependencies:

conda install -c conda-forge gym==0.21.0 numpy==1.21.4 matplotlib==3.5.0 scikit-learn==1.0.1 scipy==1.7.3 prettytable==2.4.0
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

Alternatively, you can use the provided `environment.yml` file:

## Running the Model

To run the model, use the following command structure:

Example:

python main.py hhphphphphhhhphppphppphpppphppphpp 42 50mer-DQN-Seed42-600K 10000 0 &


Parameters:
- `seq`: The HP sequence
- `seed`: Random seed for reproducibility
- `name for directory`: Output directory name
- `number of episodes`: Number of training episodes
- `early stop`: Early stopping flag (0 or 1)

## Sample Output

Below is a sample conformation output from the model:

<img src="./conformation.png" width="500">

## Additional Information

[Add any additional information, explanations, or documentation about your project here.]
