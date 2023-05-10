# 
# Nathan Lay
# AI Resource at National Cancer Institute
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
# 

import os
import sys
import uuid
import time
import SimpleITK as sitk

def _RemoveBadSlices(fileNames):
    fileSizes = [ os.path.getsize(fileName) for fileName in fileNames ]

    maxBadSlices = max(1, int(0.25*len(fileNames)))

    maxSize = max(fileSizes)

    newFileNames = [ fileName for fileName, size in zip(fileNames, fileSizes) if 3*size >= maxSize ]

    return newFileNames if len(fileNames) - len(newFileNames) <= maxBadSlices else fileNames

def LoadDicomImage(path, seriesUID = None, dim = None, dtype = None):
    if not os.path.exists(path):
        return None

    reader2D = sitk.ImageFileReader()
    reader2D.SetImageIO("GDCMImageIO")
    reader2D.SetLoadPrivateTags(True)

    if dtype is not None:
        reader2D.SetOutputPixelType(dtype)

    if dim is None: # Guess the dimension by the path
        dim = 2 if os.path.isfile(path) else 3

    if dim == 2:
        reader2D.SetFileName(path)

        try:
            return reader2D.Execute()
        except:
            return None

    if os.path.isfile(path):
        reader2D.SetFileName(path)

        try:
            reader2D.ReadImageInformation()
            seriesUID = reader2D.GetMetaData("0020|000e").strip()
        except:
            return None

        path = os.path.dirname(path)
        
    fileNames = []

    if seriesUID is None or seriesUID == "":
        allSeriesUIDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(path)

        if len(allSeriesUIDs) == 0:
            return None

        for tmpUID in allSeriesUIDs:
            tmpFileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, tmpUID)        
            
            if len(tmpFileNames) > len(fileNames):
                seriesUID = tmpUID
                fileNames = tmpFileNames # Take largest series
    else:
        fileNames = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path, seriesUID)

    fileNames = _RemoveBadSlices(fileNames)

    if len(fileNames) == 0: # Huh?
        return None

    reader3D = sitk.ImageSeriesReader()
    reader3D.SetImageIO("GDCMImageIO")
    reader3D.SetFileNames(fileNames)
    reader3D.SetLoadPrivateTags(True)
    reader3D.SetMetaDataDictionaryArrayUpdate(True)

    #reader3D.SetOutputPixelType(sitk.sitkUInt16)

    if dtype is not None:
        reader3D.SetOutputPixelType(dtype)

    try:
        image = reader3D.Execute()
    except:
        return None

    # Check if meta data is available!
    # Copy it if it is not!
    if not image.HasMetaDataKey("0020|000e"):
        for key in reader3D.GetMetaDataKeys(0): # Was 1
            image.SetMetaData(key, reader3D.GetMetaData(0, key)) # Was (1, key)

    return image

def SaveDicomImage(image, path, compress=True):
    # Implement pydicom's behavior
    def GenerateUID(prefix="1.2.826.0.1.3680043.8.498."):
        if not prefix:
            prefix = "2.25."
    
        return str(prefix) + str(uuid.uuid4().int)

    if image.GetDimension() != 2 and image.GetDimension() != 3:
        raise RuntimeError("Only 2D or 3D images are supported.")

    if not image.HasMetaDataKey("0020|000e"):
        print("Error: Reference meta data does not appear to be DICOM?", file=sys.stderr)
        return False

    writer = sitk.ImageFileWriter()
    writer.SetImageIO("GDCMImageIO")
    writer.SetKeepOriginalImageUID(True)
    writer.SetUseCompression(compress)

    newSeriesUID = GenerateUID()

    if image.GetDimension() == 2:
        writer.SetFileName(path)

        imageSlice = sitk.Image([image.GetSize()[0], image.GetSize()[1], 1], image.GetPixelID(), image.GetNumberOfComponentsPerPixel())
        imageSlice.SetSpacing(image.GetSpacing())

        imageSlice[:,:,0] = image[:]

        # Copy meta data
        for key in image.GetMetaDataKeys():
            imageSlice.SetMetaData(key, image.GetMetaData(key))

        newSopInstanceUID = GenerateUID()

        imageSlice.SetMetaData("0020|000e", newSeriesUID)
        imageSlice.SetMetaData("0008|0018", newSopInstanceUID)
        imageSlice.SetMetaData("0008|0003", newSopInstanceUID)

        try:
            writer.Execute(image)
        except:
            return False

        return True

    if not os.path.exists(path):
        os.makedirs(path)

    for z in range(image.GetDepth()):
        newSopInstanceUID = GenerateUID()

        imageSlice = sitk.Image([image.GetSize()[0], image.GetSize()[1], 1], image.GetPixelID(), image.GetNumberOfComponentsPerPixel())

        imageSlice[:] = image[:,:,z]
        imageSlice.SetSpacing(image.GetSpacing())

        # Copy meta data
        for key in image.GetMetaDataKeys():
            imageSlice.SetMetaData(key, image.GetMetaData(key))

        # Then write new meta data ...
        imageSlice.SetMetaData("0020|000e", newSeriesUID)
        imageSlice.SetMetaData("0008|0018", newSopInstanceUID)
        imageSlice.SetMetaData("0008|0003", newSopInstanceUID)

        # Instance creation date and time
        imageSlice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        imageSlice.SetMetaData("0008|0013", time.strftime("%H%M%S"))

        # Image number
        imageSlice.SetMetaData("0020|0013", str(z+1))

        position = image.TransformIndexToPhysicalPoint((0,0,z))

        # Image position patient
        imageSlice.SetMetaData("0020|0032", f"{position[0]}\\{position[1]}\\{position[2]}")

        # Slice location
        imageSlice.SetMetaData("0020|1041", str(position[2]))

        # Spacing
        imageSlice.SetMetaData("0018|0050", str(image.GetSpacing()[2]))
        imageSlice.SetMetaData("0018|0088", str(image.GetSpacing()[2]))

        imageSlice.EraseMetaData("0028|0106")
        imageSlice.EraseMetaData("0028|0107")

        slicePath = os.path.join(path, f"{z+1}.dcm")
        writer.SetFileName(slicePath)

        try:
            writer.Execute(imageSlice)
        except:
            print(f"Error: Failed to write slice '{slicePath}'.", file=sys.stderr)
            return False

    return True

