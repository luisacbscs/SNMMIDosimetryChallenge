import itk
import pydicom
import os
import numpy as np
import torch


def read_dicom_series(series_path, pixel_type=itk.F, dimension=3):
    # Set up the image readers with their type
    ImageType = itk.Image[pixel_type, dimension]

    # Using GDCMSeriesFileNames to generate the names of
    # DICOM files.
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetDirectory(series_path)

    # Get the names of files
    fileNames = namesGenerator.GetInputFileNames()

    # Set up the image series reader using GDCMImageIO
    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    dicomIO.LoadPrivateTagsOn()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)

    reader.UpdateLargestPossibleRegion()    # only works in function with this

    return reader.GetOutput()


def save_image_from_array(img_arr, img_meta, output_file="output.nii.gz"):

    img_itk = itk.image_from_array(np.asarray(img_arr))
    for k, v in img_meta.items():
        img_itk[k] = v
    itk.imwrite(img_itk, output_file)


def get_acquisition_datetime(series_path):

    metadata = pydicom.filereader.dcmread(os.path.join(series_path, sorted(os.listdir(series_path))[0]))

    acquisition_date = str(metadata[0x0008, 0x0022].value)     # Format YYYYMMDD
    acquisition_time = str(metadata[0x0008, 0x0032].value).split('.')[0]     # Format HHMMSS.(mm)

    return acquisition_date, acquisition_time


# FUNCTION TO RESAMPLE ITK VOLUME TO GIVEN SPACING (ORDER OF SPACING ARRAY: X, Y, Z - ITK FORMAT)
def resample_volume(volume, new_spacing, interpolation_mode="nearestneighbour"):

    if interpolation_mode == "nearestneighbour":
        interpolator = itk.NearestNeighborInterpolateImageFunction
    elif interpolation_mode == "linear":
        interpolator = itk.LinearInterpolateImageFunction

    if isinstance(volume, str):
        volume = itk.imread(volume)

    original_spacing = itk.spacing(volume)
    original_origin = itk.origin(volume)
    original_size = itk.size(volume)
    original_direction = volume.GetDirection()

    if original_spacing != new_spacing:

        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in
                    zip(original_size, original_spacing, new_spacing)]

        return itk.resample_image_filter(
            volume,
            interpolator=interpolator.New(volume),
            size=new_size,
            output_spacing=new_spacing,
            output_origin=original_origin,
            output_direction=original_direction
        )

    else:
        return volume


# Converts SPECT from concentration (assuming it's in Bq/mL) to activity; performs convolution with S-value kernel
# (kernel loaded from file downloaded from https://www.medphys.it/down_svoxel.asp); returns total dose rate(mGy/s) in
# VOI (+ dose rate map if requested)
def get_dose_rate(spect_path, mask_path, voi_label, return_dose_map=False, kernel_path="177Lu2.21mmsoft.txt"):

    # Loading kernel into tensor
    with open(kernel_path, "r+") as f:
        lines = f.readlines()
    voxel_size = float(lines[0].strip().split('-')[1].replace('mm', ''))
    voxel_volume = voxel_size * voxel_size * voxel_size     # voxel volume in cubic millimeters
    voxel_volume = voxel_volume * 0.001       # voxel volume in cubic centimeters or milliliters
    lines = lines[2:]

    krnl_shape = (int(lines[-1].split('\t')[0]) + 1,
                  int(lines[-1].split('\t')[1]) + 1,
                  int(lines[-1].split('\t')[2]) + 1)
    krnl = np.zeros(krnl_shape)

    for line in lines:
        p = line.split('\t')
        i, j, k, v = int(p[0]), int(p[1]), int(p[2]), float(p[3].replace('\n', ''))
        krnl[i, j, k] = v

    krnl = np.pad(krnl, ((krnl_shape[0] - 1, 0), (krnl.shape[1] - 1, 0), (krnl_shape[2] - 1, 0)), 'reflect')
    krnl = torch.Tensor(krnl)

    # Read and resample SPECT image
    img = resample_volume(spect_path, new_spacing=np.array([voxel_size, voxel_size, voxel_size]))

    arr = np.array(img) * (voxel_volume / np.power(10, 6))      # From concentration - Bq/mL - to activity - MBq

    # save_image_from_array(arr, dict(img), f"{spect_path.split(os.sep)[-1].replace('.nii.gz', '_MBq.nii.gz')}")

    tsr = torch.Tensor(arr)

    # Read and resample mask and get the VOI specified by voi_label
    msk = np.asarray(resample_volume(mask_path, new_spacing=np.array([voxel_size, voxel_size, voxel_size]),
                                     interpolation_mode='linear'))

    voi_seg = np.zeros(msk.shape)
    voi_seg[msk == voi_label] = 1
    voi_tsr = torch.Tensor(voi_seg)

    # Perform convolution: MBq * (mGy/(MBq * s)) --> mGy/s
    # torch tensor format: (batch_size, in_channels, height, width)
    dose_rate_tsr = torch.conv3d(tsr.unsqueeze(0).unsqueeze(0), krnl.unsqueeze(0).unsqueeze(0),
                                 padding='same').squeeze(0).squeeze(0)

    voi_dose_rate = torch.mul(dose_rate_tsr, voi_tsr)

    # save_image_from_array(np.asarray(dose_rate_tsr), dict(img), "whole-body_dose_rate.nii.gz")
    # save_image_from_array(np.asarray(voi_dose_rate), dict(img), f"voi_label{voi_label}_dose_rate.nii.gz")

    total_dose_rate_voi = float(torch.mean(voi_dose_rate[voi_dose_rate > 0]))

    if return_dose_map:
        return total_dose_rate_voi, np.asarray(voi_dose_rate)
    else:
        return total_dose_rate_voi
