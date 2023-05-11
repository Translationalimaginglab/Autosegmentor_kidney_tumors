## Download Latest Model 

You can download the latest model from this Google Drive link. Once downloaded, place the model file in the same directory as the other project files.

```sh 
https://drive.google.com/file/d/1Z3fB3aZSwSNSnIpX_6bYvew5nkGxENgt/view?usp=sharing
```


# Kidney and Renal Mass Segmentation

This project provides Python scripts for automatic kidney and renal mass segmentation in medical images, specifically CT and MRI scans. It includes a command-line interface (CLI) and a graphical user interface (GUI) for a more user-friendly experience.


## Dependencies

The following libraries are required to run the project:

- Python (3.8, 3.9, or 3.10)
- SimpleITK
- Torch
- MONAI
- SciPy
- OpenCV-Python
- PySimpleGUI
- Einops

## Usage

### Command-line Interface (CLI)

The CLI version can be run using the `segment_kidneys_and_masses.py` script. The script takes the following arguments:

- `--precontrast-path`: Path to the precontrast image
- `--20second-path`: Path to the 20-second delay contrast image
- `--70second-path`: Path to the 70-second delay contrast image
- `--3min-path`: Path to the 3-minute delay contrast image
- `output_image_path`: (Optional) Path to the output segmentation. If not provided, the output will be saved with the same name as the input, appended with "_seg".

For example:

```bash
python segment_kidneys_and_masses.py --precontrast-path "path/to/precontrast" --20second-path "path/to/arterial" --70second-path "path/to/venous" --3min-path "path/to/excretory" "path/to/output"
```
### Graphical User Interface (GUI)

The GUI version can be run using the `AutoSegmentationGUI3DHF.py` script. The interface allows users to select the input directories for the precontrast, arterial, venous, and excretory images, as well as the output directory. Users can also choose the modality, pathology (if known), and a model file (for new models).

After running the segmentation, the GUI allows users to browse the segmented images.

To run the GUI, simply execute the following command:

```sh
python AutoSegmentationGUI3DHF.py
```
Acknowledgments
This project was developed based on the expertise provided by medical professionals and researchers in the field of radiology and medical image analysis. Their valuable input and guidance helped create a reliable and efficient tool for kidney and renal mass segmentation.


## Acknowledgments
This project was developed based on the expertise provided by medical professionals and researchers in the field of radiology and medical image analysis. Their valuable input and guidance helped create a reliable and efficient tool for kidney and renal mass segmentation.

## References

Please cite the following articles when using this project:
1. Yazdian Anari, Pouria, et al. "Automatic segmentation of clear cell renal cell tumors, kidney, and cysts in patients with von Hippel-Lindau syndrome using U-net architecture on magnetic resonance images." arXiv preprint arXiv:2301.02538 (2023).

2. Lay, Nathan, et al. "Deep learning‐based decision forest for hereditary clear cell renal cell carcinoma segmentation on MRI." Medical Physics (2023).

other references are under final review. 

