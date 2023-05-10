Kidney and Renal Mass Segmentation

This repository contains a set of Python scripts for automatic kidney and renal mass segmentation in medical images, specifically CT and MRI scans. The project includes a command-line interface (CLI) and a graphical user interface (GUI) for a more user-friendly experience.

Dependencies
The following libraries are required to run the project:

Python (3.8, 3.9, or 3.10)

SimpleITK

Torch

MONAI

SciPy

OpenCV-Python

PySimpleGUI

Einops



Usage

Command-line Interface (CLI)

The CLI version can be run using the segment_kidneys_and_masses.py script. The script takes the following arguments:

--precontrast-path: Path to the precontrast image

--20second-path: Path to the 20-second delay contrast image

--70second-path: Path to the 70-second delay contrast image

--3min-path: Path to the 3-minute delay contrast image

output_image_path: (Optional) Path to the output segmentation. If not provided, the output will be saved with the same name as the input, appended with "_seg".
For example:
python segment_kidneys_and_masses.py --precontrast-path "path/to/precontrast" --20second-path "path/to/arterial" --70second-path "path/to/venous" --3min-path "path/to/excretory" "path/to/output"

Graphical User Interface (GUI)

The GUI version can be run using the segmentation_gui.py script. The interface allows users to select the input directories for the precontrast, arterial, venous, and excretory images, as well as the output directory. Users can also choose the modality, pathology (if known), and a model file (for new models).

After running the segmentation, the GUI allows users to browse the segmented images.

To run the GUI, simply execute the following command:

python segmentation_gui.py


Acknowledgments

This project was developed based on the expertise provided by medical professionals and researchers in the field of radiology and medical image analysis. Their valuable input and guidance helped create a reliable and efficient tool for kidney and renal mass segmentation in clinical practice.
