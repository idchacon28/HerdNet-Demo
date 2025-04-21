# HerdNet 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Code for paper "[From Crowd to Herd Counting: How to Precisely Detect and Count African Mammals using Aerial Imagery and Deep Learning?](https://doi.org/10.1016/j.isprsjprs.2023.01.025)"

## Model Architecture
![](https://i.imgur.com/kevmlhV.png)

## Detection Examples
![](https://i.imgur.com/MCZWn8Z.jpg)

## License
HerdNet is available under the [`MIT License`](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/LICENSE.md) and is thus open source and freely available. For a complete list of package dependencies with copyright and license info, please look at the file [`packages.txt`](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/packages.txt)

## Pretrained Models
Models were trained separatly for each of the two datasets. These pre-trained models follow the ([`CC BY-NC-SA-4.0`](https://creativecommons.org/licenses/by-nc-sa/4.0/)) license and are available for academic research purposes only, no commercial use is permitted.

| Model   | Params | Dataset                                                      | Environment | Species                                          | F1score | MAE¹ | RMSE² |  AC³  |                                           Download                                           |
| ------- |:------:| ------------------------------------------------------------ | ---- | ------------------------------------------------ |:-------:|:----:|:-----:|:-----:|:--------------------------------------------------------------------------------------------:|
| HerdNet |  18M   | Ennedi 2019                                                  | Desert, xeric shrubland and grassland | Camel, donkey, sheep and goat                    |  73.6%  | 6.1  |  9.8  | 15.8% | [PTH file](https://dataverse.uliege.be/file.xhtml?fileId=28087&version=1.0) |
| HerdNet |  18M   | [Delplanque et al. (2022)](https://doi.org/10.58119/ULG/MIRUU5) | Tropical forest, savanna, tropical shrubland and grassland | Buffalo, elephant, kob, topi, warthog, waterbuck |  83.5%  | 1.9  |  3.6  | 7.8%  | [PTH file](https://dataverse.uliege.be/file.xhtml?fileId=28088&version=1.0) |

¹MAE, Mean Absolute Error; ²RMSE, Root Mean Square Error; ³AC, Average Confusion between species.

Note that these metrics have been computed on full-size test images.

## Installation
Create and activate the conda environment
```console
conda env create -f environment.yml
conda activate herdnet
```

Install the code
```console
python setup.py install
```

## Dataset Format
A CSV file which must contain the header **`images,x,y,labels`** for points**. Each row should represent one annotation, with at least, the image name (``images``), the object location within the image (`x`, `y`) for points and its label (`labels`):

Point dataset:
```csv
images,x,y,labels
Example.JPG,517,1653,2
Example.JPG,800,1253,1
Example.JPG,78,33,3
Example_2.JPG,896,742,1
...
```

An image containing *n* objects is therefore spread over *n* lines.

## Quick Start 

Set the seed for reproducibility
```python
from animaloc.utils.seed import set_seed
set_seed(9292)
```

Create point datasets
```python
import os
import albumentations as A

from animaloc.datasets import CSVDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT

patch_size = 512
overlap = 40
num_classes = 2 # including background
down_ratio = 2

train_dataset = CSVDataset(
    csv_file=os.path.normpath(os.path.join('train_patches_and_csv', 'gt.csv')),
    root_dir=os.path.normpath('train_patches_and_csv'),
    albu_transforms=[
        A.VerticalFlip(p=0.5), 
        A.Normalize(p=1.0)
    ],
    end_transforms=[MultiTransformsWrapper([
        FIDT(num_classes=num_classes, down_ratio=down_ratio),
        PointsToMask(radius=2, num_classes=num_classes, squeeze=True, down_ratio=int(patch_size // 16))
    ])]
)

val_dataset = CSVDataset(
    csv_file=os.path.normpath('test_han_livestock_to_herdnet_format.csv'),
    root_dir=os.path.normpath('test_han_livestock'),
    albu_transforms=[A.Normalize(p=1.0)],
    end_transforms=[DownSample(down_ratio=down_ratio, anno_type='point')]
)

```
Create dataloaders
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 1,
    shuffle = True
    )

val_dataloader = DataLoader(
    dataset = val_dataset,
    batch_size = 1,
    shuffle = False
    )
```
Instanciate HerdNet
```python
from animaloc.models import HerdNet
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio).to(device)
```

Define the losses for training HerdNet
```python
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss

weight = torch.ones(num_classes).to(device)
weight[0] = 0.1  # background class weight!

losses = [
    {'loss': FocalLoss(reduction='mean'), 'idx': 0, 'idy': 0, 'lambda': 1.0, 'name': 'focal_loss'},
    {'loss': CrossEntropyLoss(reduction='mean', weight=weight), 'idx': 1, 'idy': 1, 'lambda': 1.0, 'name': 'ce_loss'}
    ]

herdnet = LossWrapper(herdnet, losses=losses)
```

Train and validate HerdNet
```python
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator

work_dir = os.path.join(os.getcwd(), 'demo_herdnet')
os.makedirs(work_dir, exist_ok=True)

# HYPERPARAMETERS 
lr = 1e-4
weight_decay = 1e-3
epochs = 100

optimizer = Adam(params=herdnet.parameters(), lr=lr, weight_decay=weight_decay)

metrics = PointsMetrics(radius=5, num_classes=num_classes)

stitcher = HerdNetStitcher(
    model=herdnet, 
    size=(patch_size,patch_size), 
    overlap=overlap, 
    down_ratio=down_ratio, 
    reduction='mean'
    )

evaluator = HerdNetEvaluator(
    model=herdnet, 
    dataloader=val_dataloader, 
    metrics=metrics, 
    stitcher=stitcher, 
    work_dir=work_dir, 
    header='validation'
    )

trainer = Trainer(
    model=herdnet,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=epochs,
    evaluator=evaluator,
    work_dir=work_dir
    )

trainer.start(warmup_iters=20, checkpoints='best', select='max', validate_on='f1_score')
```

## Tools
### Creating Patches
To train a model, such as HerdNet, it is often useful to extract patches from the original full-size images, especially if you have a GPU with limited memory. To do so, you can use the `patcher.py` tool:
```console
python tools/patcher.py root height width overlap dest [-csv] [-min] [-all]
```

### Making Inference with a PTH File
You can get HerdNet detections from new images using the `infer.py` tool. To use it, you will need a `.pth` file obtained using this code, which also contains the label-species correspondence (`classes`) as well as the mean (`mean`) and std (`std`) values for normalization (see the code snippet below to add this information in your `.pth` file). This tool exports the detections in `.csv` format, the plots of the detections on the images, and thumbnails of the detected animals. All this is saved in the same folder as the one containing the images (i.e. `-root`). You can adjust the size of the thumbnails by changing the `-ts` argument (defaults to 256), the frequency of the prints by changing the `-pf` argument (defaults to 10), as well as the computing device by changing the `-device` argument (defaults to cuda).
```console
python tools/infer.py root pth [-ts] [-pf] [-device]
```

Code snippet to add the required information in the pth file: 
```python
import torch

pth_file = torch.load('path/to/the/file.pth')
pth_file['classes'] = {1:'species_1', 2:'species_2', ...}
pth_file['mean'] = [0.485, 0.456, 0.406]
pth_file['std'] = [0.229, 0.224, 0.225] 
torch.save(pth_file, 'path/to/the/file.pth')
```

## Citation
```
@article{
    title = {From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning?},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {197},
    pages = {167-180},
    year = {2023},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2023.01.025},
    url = {https://www.sciencedirect.com/science/article/pii/S092427162300031X},
    author = {Alexandre Delplanque and Samuel Foucher and Jérôme Théau and Elsa Bussière and Cédric Vermeulen and Philippe Lejeune}
    }
```
