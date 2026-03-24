# SLArch-Extended
Extended SLArch framework for split learning in B5G/6G networks. Implements participant selection under bandwidth constraints with channel-aware (SINR) and data-aware (Earth Mover's Distance) policies. Features parameterized trade-off (φ/β) between link reliability and statistical representativeness. Built on ns-3/5G-LENA.

## Genral Configuration
1. Install python >=3.9.25, pytorch >=2.5.1, torchvision >=0.20.1.
2. Donwload and unzip the entire project to your computer.
3. Inside the project's root directory, download and install the ns-3 (ns-3.42: ns-3-dev and 5g-lena v3.3.y) - find instructions in network folder.
4. Place the network code "ns3_5glena_parallel_batch.cc" into scratch subdirectory within the ns-3.
5. Get the datasets from the link: https://drive.google.com/drive/folders/1zXjNipWWMTgloCLOnvJvJ4O6V9B-jxHq?usp=drive_link and place them in "ds-15" inside the split subdirectory.

## Steps to Run the Project
1. Open the Linux terminal and navegate into the project root directory (we used gnome-terminal).
2. Run the command "python3 run_project.py" to run the project.
