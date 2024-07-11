import os
import itk
import numpy as np
import json
from utilities import read_dicom_series, get_acquisition_datetime

path_spect = "C:\\Users\\Luisa\\Documents\\SNMMIDosimetryChallenge\\SPECT-CT\\"
path_voi = "C:\\Users\\Luisa\\Documents\\SNMMIDosimetryChallenge\\VOIs\\"

out_path = "C:\\Users\\Luisa\\Documents\\SNMMIDosimetryChallenge\\clean\\"

injection_datetime = {
    'patient_4': {
        "date": '20181115',
        "time": '092200'},
    'patient_6': {
        "date": '20190515',
        "time": '095500'}   # Format: HHMMSS
}

organs = sorted(["liver", "ltkidney", "rtkidney", "spleen"])
organ_label_dict = {org: i+1 for i, org in enumerate(organs)}

img_shape = None
meta = None

for patient in os.listdir(path_spect):
    timepoints = {
        "injection": injection_datetime[patient]
    }

    for scan in os.listdir(os.path.join(path_spect, patient)):
        timepoints[scan] = {}

        if not os.path.exists(os.path.join(out_path, patient, scan)):
            os.makedirs(os.path.join(out_path, patient, scan))

        for modality in ["ct", "spect"]:
            img = read_dicom_series(os.path.join(path_spect, patient, scan, modality))

            if modality == "spect":
                img_shape = np.asarray(img).shape
                meta = dict(img)
                date, time = get_acquisition_datetime(os.path.join(path_spect, patient, scan, modality))
                timepoints[scan]["date"] = date
                timepoints[scan]["time"] = time

            img_filename = f"{patient.replace('_', '')}_{scan}_{modality}.nii.gz"
            itk.imwrite(img, os.path.join(out_path, patient, scan, img_filename))
            print(f"{img_filename} done")

        scan = scan.replace('s', 'S')

        for seg_type in ['Lesions', 'Organs']:
            path_to_segs = os.path.join(path_voi, patient, scan, 'Masks', seg_type)
            seg_arr = np.zeros(img_shape)

            N_labels = 0
            for seg in os.listdir(path_to_segs):
                msk = read_dicom_series(os.path.join(path_to_segs, seg), pixel_type=itk.UI)

                if seg_type == 'Lesions':
                    N_labels += 1
                    N = int(seg.replace('lesion', ''))
                    seg_arr = seg_arr + N * (np.asarray(msk) == 1)
                elif seg_type == 'Organs':
                    if seg in organ_label_dict.keys():
                        N_labels += 1
                        seg_arr = seg_arr + organ_label_dict[seg] * (np.asarray(msk) == 1)

            seg_arr[seg_arr > N_labels] = 0     # To avoid the effects of overlapping labels

            seg_itk = itk.image_from_array(seg_arr.astype(np.uint8))
            for k, v in meta.items():
                seg_itk[k] = v
            img_filename = f"{patient.replace('_', '')}_{scan.lower()}_{seg_type.lower()}_seg.nii.gz"
            itk.imwrite(seg_itk, os.path.join(out_path, patient, scan.lower(), img_filename))
            print(f"{img_filename} done")

    open(os.path.join(out_path, patient, "timepoints.json"), 'w+').close()
    with open(os.path.join(out_path, patient, "timepoints.json"), 'a+') as f:
        json.dump(timepoints, f, indent=4)
