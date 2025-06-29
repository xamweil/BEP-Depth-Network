 # 02-EventSimulation 
This module handles the generation of event data from grayscale `.png` images using the `ESIM` library. It also includes a script for creating timestamps needed to synchronize the images with event data generation. 
## Environment Setup 
Ensure you have the `Im2e` Conda environment set up before running the scripts. The environment setup file is named `Im2e.yml` and installs the necessary packages. 

To create the environment, use:
<br>```conda env create -f Im2e.yml ```
<br>```conda activate Im2e ``` 
## Scripts 
### 1. `createTimestamps.py` 
**Purpose:**
<br>Run the script and specifying the `--file` argument to indicate the folder path where the timestamp file will be saved, along with optional arguments for `n_frames` and `duration`.

**Usage:** 
<br>Run the script with the required output directory and filename for saving the timestamps. Parameters include:
- `--file`:  Path to the main folder containing the subdirectorie for the `Timestamps\Timestamps.txt` 
- `--n_frames`: (Optional) Number of frames for the sequence (default: 480). 
- `--duration`: (Optional) Duration of the sequence in seconds (default: 4 seconds).

**Command**
<br> ``python createTimestamps.py --file path/to/your_folder --n_frames 480 --duration 4
``

*Note:*  The output file will be saved in a `Timestamps` subdirectory within the specified `--file` path and will be named `Timestamps.txt` .

### 2. `ConvertIm2e.py` 
**Purpose:**
<br>This script generates event data from a folder of grayscale images using the [ESIM library](https://github.com/uzh-rpg/rpg_vid2e). Event data is saved in a binary format and is intended to emulate the output of an event camera. 
**Usage:** Run the script with the following argument: 
- `--file`: Path to the main folder containing subdirectories for `im\gray` images and the `Timestamps\Timestamps.txt` file

**Command**
<br>```python ConvertIm2e.py --file path/to/your_folder```

*Note:* Ensure that your folder structure matches the expected setup, with subdirectories `im\gray` containing the grayscale images and `Timestamps\Timestamps.txt` holding the timestamps.
## Summary of Workflow 
1. Use `createTimestamps.py` to generate a timestamp file if one is not already available. 
2. Generate event data using `ConvertIm2e.py`, which reads from the grayscale image folder and timestamp file, saving the output as binary event data.

