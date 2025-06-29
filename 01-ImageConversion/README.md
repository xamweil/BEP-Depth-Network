# 01-ImageConversion
This module handles the conversion of `.exr` image files to grayscale `.png` images, which is essential for generating event data in the next step. It includes an optional script for visualizing event data by creating an event video.
## Environment Setup
Ensure you have the `ImageConversion` Conda environment set up before running the scripts. The environment setup file is named `ImageConversion.yml` and installs the necessary packages.

To create the environment, use:<br>
```conda env create -f ImageConversion.yml```<br>
``` conda activate ImageConversion ``` 
## Scripts 
### 1. `exrToGrayPng.py` 
**Purpose:**<br>
This script reads `.exr` files, applies tone mapping and contrast adjustment, and saves the resulting grayscale images as `.png` files. 

**Usage:**<br>Run the script with the following command-line arguments:
- `--file`: Path to the main folder containing `im/rgb` images as input, and where `im/gray` will be created as the output folder for grayscale images.

**Command**
<br>```python exrToGrayPng.py --file path/to/your_folder ```
### 2. `generateEventVid.py` (Optional) 
**Purpose:**<br> This script is optional and used to visualize event data by creating a video representation. It is intended for users who want to see the events in a video format. This script reads binary event data and generates an `.mp4` video with each frame containing visualized positive and negative events. 

**Usage:** Run the script with the following arguments: 
- `--input_folder`: Path to the folder containing the event data file. 
- - `--event_file`: Name of the binary event data file (default: `event.bin`). 
- - `--fps`: Frames per second for the output video (default: `60`). 
- - `--height`: Height of the output video in pixels (default: `720`). 
- - `--width`: Width of the output video in pixels (default: `1280`).

**Command**
<br>```python generateEventVid.py --input_folder path/to/event_data --event_file event.bin --fps 60 --height 720 --width 1280 ``` 
<br>*Note:* This script is not required for generating the dataset, but it can be used if you want to visualize the event data. 
## Summary of Workflow 
1. Convert `.exr` images to grayscale `.png` images using `exrToGrayPng.py`. 
2. (Optional) Generate a video from event data using `generateEventVid.py` if a visual representation of events is desired.