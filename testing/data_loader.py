import monai
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.croppad.dictionary import DivisiblePadd
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.data.utils import pad_list_data_collate
import pandas as pd
import os

test_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        # AddChanneld("img", "label"),
        EnsureChannelFirstd(keys=["img"]),
        DivisiblePadd(
            [
                "img",
            ],
            16,
        ),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
    ]
)


def load_data_test(
    t1w_csv, seg_mask_csv, age_csv_path, brain_mask_csv, batch, root_dir
):
    # testing on CC359

    file_name = "shuff_files_camcan.csv"
    files = os.path.join(root_dir, file_name)
    shuff_data = pd.read_csv(files)
    imgs_list = list(shuff_data["imgs"])
    age_labels = list(shuff_data["age"])

    # only for camcan
    length = len(imgs_list)
    print(length)
    test = int(0.85 * length)

    imgs_list = imgs_list[test:]
    age_labels = age_labels[test:]

    filenames_test = [
        {"img": x, "age_label": z} for (x, z) in zip(imgs_list, age_labels)
    ]

    # print('filenames train', filenames_train)
    ds_test = monai.data.Dataset(filenames_test, test_transforms)
    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
    )

    return ds_test, test_loader
