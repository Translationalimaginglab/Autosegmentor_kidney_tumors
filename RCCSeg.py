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

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import monai
import numpy as np
import SimpleITK as sitk
from Net import Net
from pathlib import Path
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    RandCropByLabelClassesd,
    RandFlipd,
    SpatialPadd,
)
#from DiceLoss import DiceLoss

def add_weight_decay(model, weightDecay):
    decay = []
    noDecay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".thresholds") or name.endswith(".ordinals"):
            noDecay.append(param)
        else:
            decay.append(param)

    return [{ "params": noDecay, "weight_decay": 0.0 }, { "params": decay, "weight_decay": weightDecay }]

def load_list(path):
    if isinstance(path, str) or isinstance(path, Path):
        with open(path, mode="rt", newline="") as f:
            return [ line.strip() for line in f if len(line.strip()) > 0 ]

    assert hasattr(path, "__iter__")

    return path

def strip_ext(path):
    if path.lower().endswith(".nii.gz"):
        return path[:-7]

    return os.path.splitext(path)[0]

def load_label_map(label_path):
    label_map = dict()

    with open(label_path, mode="rt", newline="") as f:
        for line in f:
            line = line.strip().split("#")[0]

            tokens = [ token for token in line.split(" ") if len(token) > 0 ]

            if len(tokens) < 8:
                continue

            label_name = " ".join(tokens[7:]).lower()
            label = int(tokens[0])

            if label_name.startswith('"'):
                label_name = label_name[1:-1]

            if label_name == "clear label" or label_name.startswith("label"):
                continue

            new_label = -1

            if "kid" in label_name:
                new_label = 1
            elif "cy" in label_name:
                new_label = 2
            elif label_name.startswith("rk") or label_name.startswith("lk"):
                new_label = 3
            elif "un" in label_name:
                new_label = -1
            else:
                print(f"Error: Unnamed label '{label_name}' with label value {label}: {label_path}", flush=True, file=sys.stderr)

            label_map[label] = new_label

    return label_map

class LoadMask(LoadImage):
    def __call__(self, data):
        label_path = strip_ext(data) + ".txt"

        data = super().__call__(data)

        assert os.path.exists(label_path)

        mask = data
        new_mask = torch.zeros_like(mask)

        label_map = load_label_map(label_path)

        for label in torch.unique(mask):
            if label > 0 and int(label) not in label_map:
                print(f"Error: Undescribed label with value {label}: {label_path}", flush=True, file=sys.stderr)
                new_mask[mask == label] = -1

        for label, new_label in label_map.items():
            new_mask[mask == label] = new_label

        return monai.data.MetaTensor(new_mask, meta=mask.meta)

class LoadMaskd(LoadImaged):
    def __call__(self, data):
        d = dict(data)
        label_paths = [ strip_ext(d[key]) + ".txt" for key in self.key_iterator(d) ]

        d = super().__call__(d)

        for label_path, key in zip(label_paths, self.key_iterator(d)):
            assert os.path.exists(label_path)

            mask = d[key]
            new_mask = torch.zeros_like(mask)

            label_map = load_label_map(label_path)

            for label in torch.unique(mask):
                if label > 0 and int(label) not in label_map:
                    print(f"Error: Undescribed label with value {label}: {label_path}", flush=True, file=sys.stderr)
                    new_mask[mask == label] = -1

            for label, new_label in label_map.items():
                new_mask[mask == label] = new_label

            d[key] = monai.data.MetaTensor(new_mask, meta=mask.meta)

        return d

class KeepKeys(monai.transforms.MapTransform):
    def __call__(self, data):
        d = dict()

        for key in self.key_iterator(data):
            d[key] = data[key]

        return d

class RandSelectAnnotatedHalfd(monai.transforms.Randomizable, monai.transforms.MapTransform):
    small = 7**3

    def __init__(self, label_key="label", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key

    def randomize(self, data):
        self._side = self.R.randint(low=0,high=2)

    def __call__(self, data):
        d = dict(data)
        mask = d[self.label_key]
        halfX = mask.shape[-3]//2

        right_mask = mask[..., :halfX, :, :]
        left_mask = mask[..., halfX:, :, :]

        right_side = (right_mask > 0).sum() > self.small
        left_side = (left_mask > 0).sum() > self.small

        assert left_side or right_side

        if right_side and left_side:
            self.randomize(data)

            if self._side == 0:
                left_side=False
            else:
                right_side=False

        if right_side:
            for key in self.key_iterator(d):
                d[key] = d[key][..., :halfX, :, :]
        else:
            for key in self.key_iterator(d):
                d[key] = d[key][..., halfX:, :, :]

        return d

class SelectAnnotatedHalvesd(monai.transforms.MapTransform):
    small = 7**3

    def __init__(self, label_key="label", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        mask = d[self.label_key]
        halfX = mask.shape[-3]//2

        right_mask = mask[..., :halfX, :, :]
        left_mask = mask[..., halfX:, :, :]

        right_side = (right_mask > 0).sum() > self.small
        left_side = (left_mask > 0).sum() > self.small

        assert left_side + right_side > 0

        if right_side + left_side == 2:
            pass # Nothing to do
        elif right_side:
            for key in self.key_iterator(d):
                d[key] = d[key][..., :halfX, :, :]
        else:
            for key in self.key_iterator(d):
                d[key] = d[key][..., halfX:, :, :]

        return d

def connected_components(mask):
    dtype = mask.dtype

    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    mask = (mask > 0).squeeze(dim=0).type(torch.int16)

    mask = sitk.GetImageFromArray(mask.numpy(), isVector=False)
    mask = ccFilter.Execute(mask)

    mask = sitk.GetArrayViewFromImage(mask).astype(np.int16)

    mask = torch.from_numpy(mask).unsqueeze(dim=0).clone().type(dtype)

    return mask, ccFilter.GetObjectCount()

def kidney_cc_analysis(mask, kidney_label=1, mask_is_half=False):
    if not (mask == kidney_label).any():
        return mask.fill_(0)

    if not mask_is_half:
        halfX = mask.shape[-3] // 2

        rightMask = mask[..., :halfX, :, :]
        leftMask = mask[..., halfX:, :, :]

        rightMask = kidney_cc_analysis(rightMask, kidney_label=kidney_label, mask_is_half=True)
        leftMask = kidney_cc_analysis(leftMask, kidney_label=kidney_label, mask_is_half=True)

        mask[..., :halfX, :, :] = rightMask
        mask[..., halfX:, :, :] = leftMask

        return mask

    ccMask, objCount = connected_components(mask)

    if objCount == 0:
        return mask

    ccSizes = [ (label, (ccMask == label).sum()) for label in range(1,objCount+1) ]
    largestLabel = max(ccSizes, key=lambda pair : pair[1])[0]
    
    if kidney_label not in torch.unique(mask[ccMask == largestLabel]):
        return mask.fill_(0)

    mask[ccMask != largestLabel] = 0

    return mask

class KidneyCCAnalysisd(monai.transforms.MapTransform):
    def __init__(self, kidney_label=1, mask_is_half=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kidney_label = kidney_label
        self.mask_is_half = mask_is_half

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            d[key] = kidney_cc_analysis(d[key], kidney_label=self.kidney_label, mask_is_half=self.mask_is_half)

        return d

class ThresholdCystsd(monai.transforms.MapTransform):
    def __init__(self, image_key="image", label_key="label"):
        super().__init__(keys=[image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)

        image = d[self.image_key]
        mask = d[self.label_key].squeeze(dim=0)

        mask[torch.logical_and(image[0, ...] < 253, mask == 1)] = -1

        for c in range(image.shape[0]):
            mask[image[c, ...] < -188] = -1

        d[self.image_key] = image
        d[self.label_key] = mask.unsqueeze(dim=0)

        return d

class Contract(monai.transforms.Transform):
    def __init__(self, window):
        self.window = window        

    def __call__(self, data):
        from HingeTree import contract
        if not isinstance(data, monai.data.MetaTensor):
            data = monai.data.MetaTensor(data)
        # Make a batch of size 1
        return monai.data.MetaTensor(contract(data.cpu()[None, ...], self.window).squeeze(dim=0), meta=data.meta) 

class Contractd(monai.transforms.MapTransform):
    def __init__(self, window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window

    def __call__(self, data):
        from HingeTree import contract
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = monai.data.MetaTensor(contract(d[key].cpu()[None, ...], self.window).squeeze(dim=0), meta=d[key].meta)

        return d

class Expand(monai.transforms.Transform):
    def __call__(self, data):
        from HingeTree import expand
        if not isinstance(data, monai.data.MetaTensor):
            data = monai.data.MetaTensor(data)
        # Make a batch of size 1
        return monai.data.MetaTensor(expand(data.cpu()[None, ...]).squeeze(dim=0), meta=data.meta)

class Expandd(monai.transforms.MapTransform):
    def __call__(self, data):
        from HingeTree import expand
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = monai.data.MetaTensor(expand(d[key].cpu()[None, ...]).squeeze(dim=0), meta=d[key].meta)

        return d

def get_roi_1d(size, modulus):
    remainder = size % modulus

    begin = remainder // 2
    end = begin + size - remainder

    return begin, end

def crop_modulo_n(data, modulus):
    xbegin, xend = get_roi_1d(data.shape[-3], modulus)
    ybegin, yend = get_roi_1d(data.shape[-2], modulus)
    zbegin, zend = get_roi_1d(data.shape[-1], modulus)

    return data[..., xbegin:xend, ybegin:yend, zbegin:zend]

def pad_modulo_n(data, shape, modulus):
    xbegin, xend = get_roi_1d(shape[-3], modulus)
    ybegin, yend = get_roi_1d(shape[-2], modulus)
    zbegin, zend = get_roi_1d(shape[-1], modulus)

    newData = torch.zeros(list(data.shape[:-3]) + list(shape[-3:]), dtype=data.dtype, device=data.device)
    newData[..., xbegin:xend, ybegin:yend, zbegin:zend] = data

    return newData

class KidneyMotionAugmentationd(monai.transforms.Randomizable, monai.transforms.MapTransform):
    def __init__(self, label_key="label", shift_in_pixels=4, value=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key
        self.value = value
        self.shift_in_pixels = shift_in_pixels

    def randomize(self, data):
        max_channels = 0

        for key in self.key_iterator(data):
            max_channels = max(data[key].shape[0], max_channels)

        self._shift = self.R.randint(size=max_channels, low=-self.shift_in_pixels, high=self.shift_in_pixels+1)
        self._shift[0] = 0 # No shift for first channel

    def __call__(self, data):
        self.randomize(data)

        d = dict(data)

        label = d[self.label_key]

        for key in self.key_iterator(d):
            image = d[key]

            for c in range(image.shape[0]):
                if self._shift[c] == 0:
                    continue

                beginZ = max(0, self._shift[c])
                endZ = min(image.shape[-1], self._shift[c] + image.shape[-1])

                image[c, :, :, beginZ:endZ] = image[c, :, :, 0:(endZ-beginZ)]
                image[c, :, :, :beginZ] = self.value
                image[c, :, :, endZ:] = self.value

                # Ignore these regions
                label[..., :beginZ] = self.value
                label[..., endZ:] = self.value

        return d

class RandCropKidneyd(monai.transforms.Randomizable, monai.transforms.MapTransform):
    margin=0.1

    def __init__(self, shape, label_key="label", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape
        self.label_key = label_key

    def randomize(self, data):
        label = data[self.label_key]

        indices = torch.argwhere(label > 0).type(torch.float32)
        boxLower = indices.quantile(0.05, dim=0)[-3:].type(torch.long)
        boxUpper = indices.quantile(0.95, dim=0)[-3:].type(torch.long)+1

        boxSize = boxUpper-boxLower
        marginSize = torch.round(self.margin*boxSize).type(torch.long)

        boxLower -= marginSize
        boxUpper += marginSize

        boxLower = torch.maximum(torch.tensor(0), boxLower)
        boxUpper = torch.minimum(torch.tensor(label.shape[-3:]), boxUpper)

        boxSize = boxUpper-boxLower

        assert torch.Size(self.shape) > torch.Size(boxSize)

        lower = boxUpper - torch.tensor(self.shape)
        lower = torch.maximum(torch.tensor(0), lower)

        self._begin = torch.tensor(self.R.randint(low=lower, high=boxLower+1))
        self._end = torch.minimum(self._begin + torch.tensor(self.shape), torch.tensor(label.shape[-3:]))

    def __call__(self, data):
        d = dict(data)
        self.randomize(data)

        begin = self._begin
        end = self._end

        for key in self.key_iterator(d):
            d[key] = d[key][..., begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]]

        return d

# inverse() doesn't work!
class CropModuloN(monai.transforms.Transform):
    def __init__(self, modulus):
        self.modulus = modulus

    def __call__(self, data):
        newData = crop_modulo_n(data, self.modulus)

        return newData

class CropModuloNd(monai.transforms.MapTransform):
    def __init__(self, modulus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulus = modulus

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = crop_modulo_n(d[key], self.modulus)

        return d

class PadModuloN(monai.transforms.Transform):
    def __init__(self, shape, modulus):
        self.modulus = modulus
        self.shape = shape

    def __call__(self, data):
        newData = pad_modulo_n(data, self.shape, self.modulus)

        return newData

class PadModuloNd(monai.transforms.MapTransform):
    def __init__(self, shape, modulus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modulus = modulus
        self.shape = shape

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = pad_modulo_n(d[key], self.shape, self.modulus)

        return d

class RCCSeg:
    def __init__(self, data_root, device="cpu"):
        self._data_root = data_root
        self._device = device
        self._val_steps = 25
        self._save_steps = 1*self._val_steps
        self._num_classes = 4
        self._model = Net(in_channels=4, out_channels=self._num_classes).to(self._device)

    def GetDataRoot(self):
        return self._data_root

    def GetDevice(self):
        return self._device

    def SaveModel(self, file_name):
        torch.save(self._model.state_dict(), file_name)

    def LoadModel(self, file_name):
        params = torch.load(file_name, map_location=self.GetDevice())
        self._model.load_state_dict(params)

    def RunOneGrad(self, case):
        image_keys = [ "image", "image2", "image3", "image4" ]

        modulus=max(self._model.extra_outputs)
        window = self._model.extra_outputs
        
        if isinstance(case, str):
            test_imfiles = {"image": os.path.join(self.GetDataRoot(), "Images", case, "normalized_aligned.nii.gz")}
            test_extfiles = {f"image{c}": os.path.join(self.GetDataRoot(), "Images", case, f"normalized{c}_aligned.nii.gz") for c in range(2,5)}
            test_segfiles = {f"label": os.path.join(self.GetDataRoot(), "Masks", case, "mask_aligned.nii.gz")}
            test_files = {**test_imfiles, **test_extfiles, **test_segfiles}

            test_trans = Compose(
                [
                    LoadImaged(keys=image_keys),
                    LoadMaskd(keys=["label"]),
                    EnsureChannelFirstd(keys=image_keys + ["label"]),
                    ConcatItemsd(keys=image_keys, name="image"),
                    KeepKeys(keys=["image", "label"]),
                    SelectAnnotatedHalvesd(keys=["image", "label"]),
                ]
            )
        else:
            assert False

        res = test_trans(test_files)
        image = res["image"]
        mask = res["label"]

        croptrans = Compose(
            [
                CropModuloNd(keys=["image", "label"], modulus=modulus),
                Contractd(keys=["label"], window=window),
            ]
        )
        outtrans = Compose(
            [
                Activationsd(keys=["prob"], softmax=True),
                AsDiscreted(keys=["label"], argmax=True),
                Expandd(keys=["label", "prob"]),
                PadModuloNd(keys=["label", "prob", "grad", "image"], shape=image.shape, modulus=modulus),
            ]
        )

        criterion = nn.CrossEntropyLoss(ignore_index=-1).to(self.GetDevice())
        criterion2 = DiceLoss(ignore_channel=0, ignore_label=-1, p=1, smooth=0, reduction="mean").to(self.GetDevice())

        res = croptrans({"image": image, "label": mask})
        image = res["image"]
        mask = res["label"]

        for param in self._model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        self._model.eval()

        image = image.to(self.GetDevice()) # XXX: Makes image not a leaf anymore
        image.requires_grad=True
        image.retain_grad()

        mask = mask.to(self.GetDevice()).type(torch.long)

        output = self._model(image[None, ...]).squeeze(dim=0)

        loss = criterion(output[None, ...], mask) + criterion2(output[None, ...], mask)
        loss.backward()

        self._model.train()

        result = outtrans({"prob": output, "label": output, "grad": image.grad.clone().detach(), "image": image.clone().detach()})

        prob = result["prob"]
        label = result["label"]
        grad = result["grad"]
        image = result["image"]

        return label, prob, grad, image

    def RunOne(self, case):
        image_keys = [ "image", "image2", "image3", "image4" ]

        modulus=max(self._model.extra_outputs)
        
        if isinstance(case, str):
            test_imfiles = {"image": os.path.join(self.GetDataRoot(), "Images", case, "normalized_aligned.nii.gz")}
            test_extfiles = {f"image{c}": os.path.join(self.GetDataRoot(), "Images", case, f"normalized{c}_aligned.nii.gz") for c in range(2,5)}
            test_files = {**test_imfiles, **test_extfiles}

            test_trans = Compose(
                [
                    LoadImaged(keys=image_keys),
                    EnsureChannelFirstd(keys=image_keys),
                    ConcatItemsd(keys=image_keys, name="image"),
                    KeepKeys(keys=["image"]),
                ]
            )
        else:
            test_files = case
            test_trans = lambda x : {"image": x}

        image = test_trans(test_files)["image"]

        croptrans = CropModuloN(modulus=modulus)
        outtrans = Compose(
            [
                Activationsd(keys=["prob"], softmax=True),
                AsDiscreted(keys=["label"], argmax=True),
                Expandd(keys=["label", "prob"]),
                KidneyCCAnalysisd(keys=["label"]),
                PadModuloNd(keys=["label", "prob"], shape=image.shape, modulus=modulus),
            ]
        )

        image = croptrans(image)

        with torch.no_grad():
            self._model.eval()

            image = image.to(self.GetDevice())
            output = self._model(image[None, ...]).squeeze(dim=0)

            self._model.train()

            result = outtrans({"prob": output, "label": output})

            prob = result["prob"]
            label = result["label"]

        return label, prob

    def Test(self, test_list):
        test_list = load_list(test_list)

        segtrans = LoadMask(image_only=True, ensure_channel_first=True)
        onehot = Compose(
            [
                SelectAnnotatedHalvesd(keys=["pred", "label"]),
                AsDiscreted(keys=["pred", "label"], to_onehot=self._num_classes),
            ]
        )

        dice_metric = DiceMetric(include_background=True, reduction="mean_channel", get_not_nans=False, ignore_empty=False)

        allDices = dict()

        for case in test_list:
            dice_metric.reset()

            segfile = os.path.join(self.GetDataRoot(), "Masks", case, "mask_aligned.nii.gz")
            mask = segtrans(segfile)

            label, prob = self.RunOne(case)

            label[mask < 0] = 0
            mask[mask < 0] = 0

            #print(label.shape, prob.shape)

            res = onehot({"pred": label[None, ...], "label": mask[None, ...]})

            #dice_metric(y_pred=onehot(label), y=onehot(mask))
            dice_metric(y_pred=res["pred"], y=res["label"])
            metric = dice_metric.aggregate().numpy()[None, ...]
            allDices[case] = metric

            print(f"{case}: {metric}")

        return allDices

    def Train(self, train_list, snapshot_dir, val_perc=0.0, val_list=None):
        train_list = load_list(train_list)
        window = self._model.extra_outputs
        num_epochs = 1000

        if val_list is not None:
            val_list = load_list(val_list)
            print(f"Info: Loaded {len(val_list)} validation cases.")
        elif val_perc > 0:
            n = int(val_perc*len(train_list))
            val_list = train_list[:n]
            train_list = train_list[n:]

        train_imfiles = [ {"image": os.path.join(self.GetDataRoot(), "Images", case, "normalized_aligned.nii.gz")} for case in train_list ]
        train_extfiles = [ {f"image{c}": os.path.join(self.GetDataRoot(), "Images", case, f"normalized{c}_aligned.nii.gz") for c in range(2,5)} for case in train_list ]
        train_segfiles = [ {"label": os.path.join(self.GetDataRoot(), "Masks", case, "mask_aligned.nii.gz")} for case in train_list ]

        train_files = [ {**im, **ext, **seg} for im, ext, seg in zip(train_imfiles, train_extfiles, train_segfiles) ]

        image_keys = [ "image", "image2", "image3", "image4" ]

        train_trans = Compose(
            [
                LoadImaged(keys=image_keys),
                LoadMaskd(keys=["label"]),
                EnsureChannelFirstd(keys=image_keys + ["label"]),
                ConcatItemsd(keys=image_keys, name="image"),
                KeepKeys(keys=["image", "label"]),
                #SelectAnnotatedHalvesd(keys=["image", "label"]),
                RandSelectAnnotatedHalfd(keys=["image", "label"]),
                KidneyMotionAugmentationd(keys=["image"]),
                #RandCropByLabelClassesd(keys=["image", "label"], label_key="label", num_classes=4, spatial_size=[176,256,80], ratios=[1,1,1,1], allow_smaller=True),
                RandCropKidneyd(keys=["image", "label"], shape=[176,256,80]),
                ThresholdCystsd(image_key="image", label_key="label"),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                SpatialPadd(keys=["image", "label"], spatial_size=[176,256,80], mode="constant", value=-1),
                Contractd(keys=["label"], window=window),
            ]
        )

        #for d in train_files:
        #    #print(train_files[0]["image"])
        #    print(d["label"])
        #    res = train_trans(d)

        #exit()
        #print(type(res))
        #print(res[0]["image"].shape, res[0]["label"].shape)

        #train_ds = CacheDataset(data=train_files, transform=train_trans, cache_rate=1.0, num_workers=4)
        train_ds = Dataset(data=train_files, transform=train_trans)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8)

        #res = monai.utils.misc.first(train_loader)
        #print(res["image"].shape, res["label"].shape)
        #print(res["image"].shape, res["label"].shape)

        criterion = nn.CrossEntropyLoss(ignore_index=-1).to(self.GetDevice())
        criterion2 = DiceLoss(ignore_channel=0, ignore_label=-1, p=1, smooth=0, reduction="mean").to(self.GetDevice())

        baseLearningRate = 1e-2
        optimizer = optim.AdamW(self._model.parameters(), baseLearningRate)
        #optimizer = optim.AdamW(add_weight_decay(self._model, 1e-2), baseLearningRate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=baseLearningRate*0.1, max_lr=baseLearningRate, cycle_momentum=False)

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        for epoch in range(num_epochs):
            running_loss = 0.0
            count = 0

            for batch_dict in train_loader:
                xbatch = batch_dict["image"]
                ybatch = batch_dict["label"]

                xbatch = xbatch.to(self.GetDevice())
                #ybatch = ybatch.to(self.GetDevice())
                ybatch = ybatch.squeeze(dim=1).to(self.GetDevice()).type(torch.long)

                #print(torch.unique(ybatch))

                #print(torch.unique(ybatch))

                #print(xbatch.shape, ybatch.shape)
                #print(xbatch.dtype, ybatch.dtype)
                #outputs = self._model(xbatch) 
                #print(outputs.shape)
                #continue
                #exit()

                optimizer.zero_grad()

                outputs = self._model(xbatch)
                #loss = criterion2(outputs, ybatch) 
                loss = criterion(outputs, ybatch) + criterion2(outputs, ybatch)

                running_loss += loss.item()
                count += 1

                #print(f"loss = {loss.item()}")

                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                #print(xbatch.shape, xbatch.dtype)
                #print(ybatch.shape, ybatch.dtype)

            running_loss /= count

            #learningRate=scheduler.get_last_lr()
            learningRate=baseLearningRate if scheduler is None else scheduler.get_last_lr()
            print(f"Info: Epoch {epoch}: loss = {running_loss}, learning rate = {learningRate}", flush=True)

            if self._save_steps > 0 and (epoch+1) % self._save_steps == 0:
                snapshot_output = os.path.join(snapshot_dir, f"epoch_{epoch}.pt")
                print(f"Info: Saving snapshot '{snapshot_output}' ...", flush=True)
                self.SaveModel(snapshot_output)

            if val_list is not None and self._val_steps > 0 and (epoch+1) % self._val_steps == 0:
                allDices = self.Test(val_list)

                tmp = np.array(list(allDices.values()))
                mean = tmp.mean(axis=0)
                std = tmp.std(axis=0)

                print(f"Info: Epoch {epoch}: validation dice = {mean} +/- {std}", flush=True)

                dice_output = os.path.join(snapshot_dir, f"dice_stats_{epoch}.txt")

                print(f"Info: Writing dice stats '{dice_output}' ...", flush=True)

                with open(dice_output, mode="wt", newline="") as f:
                    for case, dice in allDices.items():
                        f.write(f"{case}: {dice}\n")

                    f.write(f"\nmean dice: {mean} +/- {std}\n")

        if num_epochs % self._save_steps != 0:
            snapshot_output = os.path.join(snapshot_dir, f"epoch_{num_epochs-1}.pt")
            print(f"Info: Saving final snapshot '{snapshot_output}' ...", flush=True)
            self.SaveModel(snapshot_output)

if __name__ == "__main__":
    data_root="/data/AMPrj/NiftiCombinedNew"
    train_list=os.path.join(data_root, "balanced_training0.txt")
    test_list=os.path.join(data_root, "balanced_testing0.txt")
    val_list=os.path.join(data_root, "balanced_validation0.txt")

    #model_file="/data/AIR/RCC/Models/snapshots_rcc_hingeforest_depth7_vggblock3_3d_cyclicLR_adamw_monai_nowd_split5/epoch_824.pt"

    cad = RCCSeg(data_root=data_root, device="cuda:0")
    #cad.LoadModel(model_file)

    #exit()
    #data_root="/data/AIR/RCC/NiftiEverything"
    #train_list="/data/AIR/RCC/NiftiEverything/train_randomSplit1.txt"


    #cad.Train(train_list, val_perc=0.1, snapshot_dir="/data/AIR/RCC/Models/snapshots_monai_test_20230221")
    cad.Train(train_list, val_list=val_list, snapshot_dir="/data/AIR/RCC/Models/snapshots_monai_3class_20230227")

    exit()

    cad.LoadModel("/data/AIR/RCC/Models/snapshots_monai_test_20230217/epoch_224.pt")
    allDices = cad.Test(test_list)

    tmp = torch.tensor(list(allDices.values()))
    mean = tmp.mean(dim=0)
    std = tmp.std(dim=0)

    print(f"Info: test dice = {mean} +/- {std}", flush=True)

