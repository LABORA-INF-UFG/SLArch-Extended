# SLArch-Extended
Extended SLArch framework for split learning in B5G/6G networks. Implements participant selection under bandwidth constraints with channel-aware (SINR) and data-aware (Earth Mover's Distance) policies. Features parameterized trade-off (φ/β) between link reliability and statistical representativeness. Built on ns-3/5G-LENA.

## Requirements
  1. Install the following dependencies:
     - Python ≥ 3.9.25
     - PyTorch ≥ 2.5.1
     - TorchVision ≥ 0.20.1

## General Setup
1. Donwload and unzip the entire project to your computer.
2. Inside the project's root directory, download and install ns-3.
    - Version ns-3.42 (ns-3-dev) and 5g-lena v3.3.y
    - Detailed installation instructions can be found in the "network" folder.
4. Place the network code file "ns3_5glena_parallel_batch.cc" into the "scratch" subdirectory inside the "ns-3-dev" directory.
5. Get the datasets from the following link:
    - https://drive.google.com/drive/folders/1zXjNipWWMTgloCLOnvJvJ4O6V9B-jxHq?usp=sharing
    - Place them in the "ds-15" subdirectory inside the "split" folder.

## Steps to run the project
1. Open the Linux terminal and navegate into the project root directory.
2. Run the following command:
   ```bash
    python3 run_project.py
   ```
    
