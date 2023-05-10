# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# February 2023
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
import argparse
import torch
import torch.nn as nn
import torch.autograd
import numpy as np
import SimpleITK as sitk
import scipy.optimize as optim
#from scipy.special import iv, laguerre
from common import LoadDicomImage, SaveDicomImage

def _ModifiedBessel0(x):
    z = 0.25*x*x

    if isinstance(x, torch.Tensor):
        term = torch.ones_like(z)
        theSum = term.clone()
    else:
        term = np.ones_like(z)
        theSum = term.copy()

    for k in range(1,20):
        term *= z/(k*k)
        theSum += term

    return theSum


def _ModifiedBessel1(x):
    z = 0.25*x*x

    if isinstance(x, torch.Tensor):
        term = torch.ones_like(z)
        theSum = term.clone()
    else:
        term = np.ones_like(z)
        theSum = term.copy()

    for k in range(2,20):
        term *= z/(k*(k-1))
        theSum += term

    theSum *= 0.5*x

    return theSum

# Use autograd to avoid chain of 19 graph nodes! Saves memory.
class ModifiedBessel0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _ModifiedBessel0(x)

    @staticmethod
    def backward(ctx, dx):
        x, = ctx.saved_tensors
        return _ModifiedBessel1(x)*dx

# XXX: The below don't work with large values
#def _ModifiedBessel0(x):
#    if isinstance(x, torch.Tensor):
#        return torch.special.i0(x)
#
#    return iv(0, x)
#
#def _ModifiedBessel1(x):
#    if isinstance(x, torch.Tensor):
#        return torch.special.i1(x)
#
#    return iv(1, x)

# scipy laguerre accepts only integer degrees
def _HalfLaguerre(x):
    if isinstance(x, torch.Tensor):
        exp = torch.exp
    else:
        exp = np.exp

    return exp(0.5*x)*((1.0 - x)*_ModifiedBessel0(-0.5*x) - x*_ModifiedBessel1(-0.5*x))

# NOTE: Scipy does have a Rice distribution, but it doesn't work in an obvious way with C++ RicianNormalization's formulation
class Rice:
    @staticmethod
    def Pdf(x, nu, sigma):
        if isinstance(x, torch.Tensor):
            y = torch.zeros_like(x)
            exp = torch.exp
        else:
            x = np.array(x)
            y = np.zeros_like(x)
            exp = np.exp

        mask = (x >= 0)

        y[~mask] = 0
        u = x[mask]

        sigma2 = sigma*sigma

        y[mask] = u/sigma2 * exp(-(u*u + nu*nu)/(2*sigma2)) * ModifiedBessel0.apply(u*nu/sigma2)

        return y

    @staticmethod
    def LogPdf(x, nu, sigma):
        if isinstance(x, torch.Tensor):
            y = torch.zeros_like(x)
            log = torch.log
        else:
            x = np.array(x)
            y = np.zeros_like(x)
            log = np.log

        mask = (x > 0)
        y[~mask] = 0

        u = x[mask]

        sigma2 = sigma*sigma
        #y[mask] = log(u) - log(sigma2) - (u*u + nu*nu)/(2*sigma2) + log(ModifiedBessel0(u*nu/sigma2))
        y[mask] = log(u) - log(sigma2) - (u*u + nu*nu)/(2*sigma2) + log(ModifiedBessel0.apply(u*nu/sigma2))

        return y

    @staticmethod
    def Mean(nu, sigma):
        return sigma*np.sqrt(0.5*np.pi)*_HalfLaguerre(-0.5*(nu/sigma)**2)

    @staticmethod
    def Variance(nu, sigma):
        return 2*sigma*sigma + nu*nu - 0.5*np.pi*(sigma*_HalfLaguerre(-0.5*(nu/sigma)**2))**2

def Normalize(npImg, clamp=False, verbose=False):
    dtype = npImg.dtype
    npImg = npImg.astype(np.float64)

    if verbose:
        debug = print
    else:
        debug = lambda *args, **kwargs : None

    if clamp:
        debug("Info: Clamping image to be non-negative.")
        npImg = np.maximum(0.0, npImg)

    def fun(x):
        x = nn.Parameter(torch.from_numpy(x), requires_grad=True)
        v = x[0]
        s = x[1]

        loss = -Rice.LogPdf(torch.from_numpy(npImg), v, s).mean()
        loss.backward()

        return loss.item(), x.grad.numpy().copy()

    debug("Info: Normalizing ...")

    x0 = _InitialGuess(npImg, verbose)

    bounds = [(5e-4, np.inf)]*2

    debug("Info: Running L-BFGS-B ...")
    res = optim.minimize(fun=fun, jac=True, x0=x0, bounds=bounds, method='L-BFGS-B')

    assert res.success
    debug(res)

    v = res.x[0]
    s = res.x[1]

    mean = Rice.Mean(v,s)
    std = np.sqrt(Rice.Variance(v,s))

    debug(f"nu = {v}, sigma = {s}")
    debug(f"Rice mean = {mean}, Rice std = {std}")

    scale = 100.0 if np.issubdtype(dtype, np.integer) else 1.0

    if np.issubdtype(dtype, np.unsignedinteger):
        npImg = np.maximum(0.0, scale*((npImg - mean)/std + 4.0))
    else:
        npImg = scale*(npImg - mean)/std

    return npImg.astype(dtype)

def _InitialGuess(npImg, verbose=False):
    tol = 1e-5

    if verbose:
        debug = print
    else:
        debug = lambda *args, **kwargs : None

    if np.any(npImg < 0):
        raise RuntimeError("Negative pixel value implies not Rice distributed.")

    u1 = npImg.mean()
    u2 = (npImg*npImg).mean()

    if u1 < 0 or u2 <= 0:
        raise RuntimeError("Negative moments.")

    def Objective(s):
        return s*np.sqrt(0.5*np.pi)*_HalfLaguerre(1.0 - 0.5*u2/(s*s)) - u1

    a = 1e-1
    b = np.sqrt(0.5*u2)
    fa = Objective(a)
    fb = Objective(b)
    s = 0.0

    debug("Solving for initial guess with bisection method...")
    debug(f"fa = {fa}, fb = {fb}")

    if np.abs(fb) < tol:
        s = b
    elif np.abs(fa) < tol:
        s = a
    else:
        for i in range(21):
            s = 0.5*(a + b)

            fs = Objective(s)

            debug(f"s = {s}, fs = {fs}")

            if np.abs(fs) < tol:
                break

            if (fb < 0) ^ (fs < 0):
                a = s
                fa = fs
            elif (fa < 0) ^ (fs < 0):
                b = s
                fb = fs
            else:
                break # Uhh?
                
    v = np.sqrt(2.0)*s*np.sqrt(0.5*u2/(s*s) - 1.0)

    debug("\n" + f"Initial nu = {v}, sigma = {s}")
    debug(f"Sample mean = {u1}, Rice mean = {Rice.Mean(v,s)}")
    debug(f"Sample 2nd moment = {u2}, Rice 2nd moment = {2*s*s + v*v}")

    return np.array([v,s])

def main(inputPath, outputPath, clamp=False, compress=False, outputType=None):
    inputIsSeries = os.path.isdir(inputPath)
    outputIsSeries = (len(os.path.splitext(outputPath)[1]) == 0)

    if outputIsSeries and not inputIsSeries:
        print(f"Error: Cannot save DICOM series from non-DICOM image.", file=sys.stderr)
        exit(1)

    if inputIsSeries:
        image = LoadDicomImage(inputPath)
    else:
        image = sitk.ReadImage(inputPath)

    npImage = sitk.GetArrayFromImage(image)

    if outputType is not None:
        print(f"Info: Converting to {outputType} ...")
        npImage = npImage.astype(outputType)

    npImage = Normalize(npImage, clamp=clamp, verbose=True)

    newImage = sitk.GetImageFromArray(npImage)
    newImage.CopyInformation(image)

    if outputIsSeries:
        for key in image.GetMetaDataKeys():
            newImage.SetMetaData(key, image.GetMetaData(key))

        # Prevent ITK from inverting any transform when saving
        newImage.SetMetaData("0028|1052", "0") # Intercept
        newImage.SetMetaData("0028|1053", "1") # Slope
        newImage.SetMetaData("0028|1054", "US") # US - "Unspecified"

        print(f"Info: Saving DICOM series to '{outputPath}' ...")
        SaveDicomImage(newImage, outputPath, compress=compress)
    else:
        print(f"Info: Saving image to '{outputPath}' ...")
        sitk.WriteImage(newImage, outputPath, useCompression=compress)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyRicianNormalization")
    parser.add_argument("-C", "--clamp", dest="clamp", action="store_true", default=False, help="Clamp image so that it is non-negative.")
    parser.add_argument("--output-type", dest="outputType", required=False, default=None, help="Specify output voxel type (e.g. float32, int16).")
    parser.add_argument("--compress", dest="compress", action="store_true", default=False, help="Compress image output.")
    parser.add_argument("inputPath", type=str, help="Input image path.")
    parser.add_argument("outputPath", type=str, help="Output image path. A folder path implies DICOM output.")

    args = parser.parse_args()

    main(**vars(args))

