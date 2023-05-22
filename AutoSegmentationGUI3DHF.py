# Pouria Yazdian Anari, Nathan Lay, and AMPrj team
# National Institutes of Health
# September 2021
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import sys
import subprocess
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pkg_resources

def install_packages(python_version):
    wheel_file = f"hingetree_cpp-0.0.0-cp3{python_version}-cp3{python_version}-win_amd64.whl"
    pip_install_wheel = f"pip install {wheel_file}"

    required_libraries = [
        "torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1",
        "SimpleITK",
        "monai",
        "scipy",
        "opencv-python",
        "PySimpleGUI",
        "einops",
    ]

    installed_packages = [pkg.key for pkg in pkg_resources.working_set]

    try:
        if "hingetree_cpp-0.0.0-cp38-cp38-win_amd64" not in installed_packages:
            print(f"Installing {wheel_file}...")
            subprocess.check_call(pip_install_wheel, shell=True)

        print("Checking required libraries...")
        for lib in required_libraries:
            if lib.split("==")[0] not in installed_packages:
                print(f"Installing {lib}...")
                subprocess.check_call(f"pip3 install {lib}", shell=True)
            else:
                print(f"{lib} is already installed.")
        
        print("All required packages are installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing packages: {e}")
# Get the current Python version
major, minor = sys.version_info.major, sys.version_info.minor
    
# Check if the Python version is supported and install the required packages
if (major, minor) in [(3, 8), (3, 9), (3, 10)]:
    install_packages(minor)
else:
    print("Unsupported Python version. Please use Python 3.8, 3.9, or 3.10.")
    

import PySimpleGUI as sg
import cv2


def get_image_files(folder_path):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions]
    return image_files

def resize_image(image_path, max_width, max_height):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if aspect_ratio >= 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
    else:
        new_width = width
        new_height = height

    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img

def Segment_cmd(PreContrast, Arterial,Venous,Excretory,Output):
    cmd = f'python segment_kidneys_and_masses.py --precontrast-path "{PreContrast}" --20second-path "{Arterial}" --70second-path "{Venous}" --3min-path "{Excretory}" "{Output}"'
    result = subprocess.check_output(cmd, shell=True)
    print(result.decode())  
    
if __name__ == "__main__":

    
    layout = [
        [sg.Text('Select Pre-Contrast Input Directory:')],
        [sg.Input(key='PreContrast'), sg.FolderBrowse()],
        [sg.Text('Select Arterial Input Directory:')],
        [sg.Input(key='Arterial'), sg.FolderBrowse()],
        [sg.Text('Select Venous Input Directory:')],
        [sg.Input(key='Venous'), sg.FolderBrowse()],
        [sg.Text('Select Excretory Input Directory:')],
        [sg.Input(key='Excretory'), sg.FolderBrowse()],
        [sg.Text('Select Output Directory:')],
        [sg.Input(key='output_dir'), sg.FolderBrowse()],
        [sg.Text('Select modality:')],
        [sg.Button('MRI'), sg.Button('CT'), sg.Button('MRI Kidney only'), sg.Button('CT Kidney only')],
        [sg.Text('Select Pathology (if known):')],
        [sg.Button('Angiomyolipoma'), sg.Button('Chromophobe'), sg.Button('Clear cell renal cell carcinoma'),
         sg.Button('HLRCC'), sg.Button('Hybrid'), sg.Button('Oncocytoma'), sg.Button('Papillary')],
        [sg.Text('Model File (for new models only):')],
        [sg.Input(key='model_name'), sg.FileBrowse()],
        [sg.Button('Detect'), sg.Exit()],

    ]
    
    window = sg.Window('Auto-Segmentation V.1.05', layout, resizable=True, finalize=True)
    selected_pathology = ''
    image_files = []
    current_image_idx = 0
    max_width, max_height = 600, 450
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Detect':
            if values['PreContrast'] and values['Arterial'] and values['Venous'] and values['Excretory'] and values['output_dir']:
                PreContrast = values['PreContrast']
                Arterial = values['Arterial']
                Venous = values['Venous']
                Excretory = values['Excretory']
                output_dir = values['output_dir']
                Segment_cmd(PreContrast, Arterial, Venous, Excretory, output_dir)
                print(f"Segmentation completed for {output_dir}")


        elif event in ('Angiomyolipoma', 'Chromophobe', 'Clear cell renal cell carcinoma', 'HLRCC', 'Hybrid', 'Oncocytoma', 'Papillary'):
            selected_pathology = event
            sg.popup(f'You have selected {selected_pathology} pathology.')
        elif event == 'Open Images':
            output_folder = values['output_dir']
            image_files = get_image_files(output_folder)
            if image_files:
                current_image_idx = 0
                resized_img = resize_image(image_files[current_image_idx], max_width, max_height)
                window['output_image'].update(data=cv2.imencode('.png', resized_img)[1].tobytes())
            else:
                sg.popup("No image files found in the selected folder.")
        elif event == 'Next':
            if image_files:
                current_image_idx = (current_image_idx + 1) % len(image_files)
                resized_img = resize_image(image_files[current_image_idx], max_width, max_height)
                window['output_image'].update(data=cv2.imencode('.png', resized_img)[1].tobytes())
        elif event == 'Previous':
            if image_files:
                current_image_idx = (current_image_idx - 1) % len(image_files)
                resized_img = resize_image(image_files[current_image_idx], max_width, max_height)
                window['output_image'].update(data=cv2.imencode('.png', resized_img)[1].tobytes())
    
    window.close()
