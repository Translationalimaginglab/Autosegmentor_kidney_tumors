/*
 * ----------------------------------------------------------------------------
 * Copyright (c) 2023, Nathan Lay, Pouria Yazdian, and AMPrj Lab Members
 *
 * This code has been developed by Pouria Yazdian, Nathan Lay, and members
 * of the AMPrj Lab. All rights reserved. This code is intended for
 * non-commercial use only.
 *
 * Redistribution, modification, and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions, and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions, and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ----------------------------------------------------------------------------
 */


import os
import argparse
import SimpleITK as sitk
import torch
import numpy as np
from RCCSeg import RCCSeg
from RicianNormalization import Normalize
from common import LoadDicomImage, SaveDicomImage

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def split_ext(path):
    if path.lower().endswith(".nii.gz"):
        return path[:-7], path[-7:]
    
    return os.path.splitext(path)

def get_ext(path):
    _, ext = split_ext(path)
    return ext
    
def resample(image, new_spacing=None, new_size=None, ref_img=None, interp="nearest"):
    interp_dict = { "nearest": sitk.sitkNearestNeighbor, "linear": sitk.sitkLinear, "bsplinbe": sitk.sitkBSpline }
    
    assert interp in interp_dict, f"'{interp}' not a supported interpolation mode."

    assert image.GetDimension() == 3
    # Only one of these should be set
    assert (new_spacing is not None) + (new_size is not None) + (ref_img is not None) == 1
    
    interp = interp_dict[interp]
    
    spacing = image.GetSpacing()
    size = image.GetSize()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    
    if new_spacing is not None:
        new_size = [ round((sz*sp)/nsp) for sz, sp, nsp in zip(size, spacing, new_spacing) ]
        
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
    elif new_size is not None:
        new_spacing = [ (sz*sp)/nsz for sz, sp, nsz in zip(size, spacing, new_size) ]
    
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
    else:
        resampler.SetReferenceImage(ref_img)

    new_image = resampler.Execute(image)
        
    return new_image

def main(three_min_path, precontrast_path, twenty_second_path, seventy_second_path, output_image_path=None, device="cpu", weights="epoch_724.pt", compress=False):
    path_seps = os.sep
    
    if hasattr(os, "altsep") and os.altsep is not None:
        path_seps += os.altsep

    if output_image_path is None:
        output_image_path, ext = split_ext(three_min_path)
        output_image_path = f"{output_image_path}_seg{ext}"
        
    input_image_paths = ( three_min_path, precontrast_path, twenty_second_path, seventy_second_path )
        
    input_is_dicom = any(os.path.isdir(path) for path in input_image_paths)
    output_is_dicom = (len(get_ext(output_image_path)) == 0)
    
    if output_is_dicom and not input_is_dicom:
        print(f"Error: DICOM output only possible if input is also DICOM.")
        exit(1)
        
    images = []
    
    for path in input_image_paths:
        assert os.path.exists(path), f"{path}"
        if os.path.isdir(path):
            image = LoadDicomImage(path)
        else:
            image = sitk.ReadImage(path)
            
        images.append(image)
    
    assert all(image is not None for image in images)
    
    orig_size = images[0].GetSize()
    
    new_images = []
    new_images.append(resample(images[0], new_spacing=[1.0, 1.0, 3.0]))
    
    for image in images[1:]:
        new_images.append(resample(image, ref_img=new_images[0], interp="linear"))
        
    assert all(image is not None for image in new_images)
    
    norm_images = []
    for image in new_images:
        norm_images.append(Normalize(sitk.GetArrayFromImage(image).astype(np.int16), verbose=True))
    
    assert all(image is not None for image in norm_images)
    
    # Get the image ready for MONAI
    norm_images = [ image.transpose(2,1,0)[None, ...] for image in norm_images ]
    
    norm_image = np.concatenate(tuple(norm_images), axis=0)
    
    norm_image = torch.from_numpy(norm_image).type(torch.float32)
    
    cad = RCCSeg(data_root=".", device=device)
    cad.LoadModel(weights)
    
    label, prob = cad.RunOne(norm_image)
    
    label = label.type(torch.int16).squeeze(dim=0)
    label = label.cpu().numpy().transpose(2,1,0)
    
    label = sitk.GetImageFromArray(label, isVector=False)
    label.CopyInformation(new_images[0])
    
    label = resample(label, new_size=orig_size)
    
    print(f"Info: Writing '{output_image_path}' ...")
    
    if output_is_dicom:
        dicom_image = next(iter((image for image in images if image.HasMetaDataKey("0020|000e"))))
    
        for key in dicom_image.GetMetaDataKeys():
            label.SetMetaData(key, dicom_image.GetMetaData(key))
        
        assert SaveDicomImage(label, output_image_path, compress=compress)
    else:    
        sitk.WriteImage(label, output_image_path, useCompression=compress)
        
    print("Done.")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="segment_kidneys_and_masses")
    parser.add_argument("--compress", dest="compress", action="store_true", default=False, help="Compress segmentation output.")
    parser.add_argument("--device", dest="device", type=str, default="cpu", help="PyTorch compute device.")
    parser.add_argument("--weights", dest="weights", type=str, default="epoch_724.pt", help="Path to weights for segmentation model.")
    parser.add_argument("--precontrast-path", dest="precontrast_path", required=True, type=str, help="Path to precontrast image path.")
    parser.add_argument("--20second-path", dest="twenty_second_path", required=True, type=str, help="20 second delay contrast image path.")
    parser.add_argument("--70second-path", dest="seventy_second_path", required=True, type=str, help="70 second delay contrast image path.")
    parser.add_argument("--3min-path", dest="three_min_path", required=True, type=str, help="3 minute delay contrast image path.")
    #parser.add_argument("input_image_path", type=str, help="Input image path.")
    parser.add_argument("output_image_path", type=str, nargs="?", default=None, help="Output segmentation path. A folder implies DICOM output.")
    
    args = parser.parse_args()
    
    main(**vars(args))
    
    
