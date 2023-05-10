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
