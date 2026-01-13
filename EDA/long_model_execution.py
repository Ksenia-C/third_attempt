# Usage: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python3 long_model_execution.py --data_filepath /home/kpetrenko/work/data_skin/save_dataset.pkl --model_version '/home/kpetrenko/work/models_best_30/skin' --freeze_begin --randomize_end
import eda_helpers as eda
import re
import psutil
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models_arch'))
import json
import argparse
import densenet3
# import torch._dynamo
# torch._dynamo.config.recompile_limit = 64

# Variable
DATASET_PATH = 'flwrlabs/fed-isic2019'
SAVE_LOCALLY = '/home/kpetrenko/work/data_skin/save_dataset.pkl'
LABELS = list(map(str, range(8)))
GROUP_COL = 'center'

# Args parser
parser = argparse.ArgumentParser("model learning execution with saved and newly downloaded versions")

# Add arguments
parser.add_argument('--data_filepath', help='where to take data from; None=download; Filepath=upack pkl')
parser.add_argument('--model_version', help='start model. Supported densenet121: random_download, default_download, [path to saved model weight]; Suppoerted resnet: resnet_default_downloaded', default='random_download')
parser.add_argument('--freeze_begin', action='store_true', help='freeze all except for the last layer')
parser.add_argument('--randomize_end', action='store_true', help='freeze all except for the last layer')

# Parse arguments
args = parser.parse_args()


def randomize_weights_densenet(module):
    # lookup in the package:
    # Official init from torch repo.
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.constant_(module.bias, 0)

device = torch.device('cpu')
torch.set_num_threads(min(20, psutil.cpu_count()))

if args.data_filepath is None:
    data = eda.download_dataset(DATASET_PATH, 'train')
    data['image'] = data['image'].map(lambda x: eda.resize_and_pad(x, (128, 128)))
else:
    data = eda.up_dataset(args.data_filepath)


architecuter = None
if args.model_version == 'resnet_default_downloaded':
    model = eda.models.resnet50(weights=eda.models.ResNet50_Weights.DEFAULT)
    model.fc = eda.nn.Linear(model.fc.in_features, len(LABELS))
    architecuter = "resnet"
if args.model_version == 'random_download':
    model = eda.models.densenet121(weights=None)
    model.classifier = eda.nn.Linear(model.classifier.in_features, len(LABELS))
    architecuter = "densenet"
elif args.model_version == 'default_download':
    model = eda.models.densenet121(weights="DEFAULT")
    model.classifier = eda.nn.Linear(model.classifier.in_features, len(LABELS))
    architecuter = "densenet"
elif args.model_version == 'random_efficient':
    model = eda.models.efficientnet_b3(weights=None)
    architecuter = "efficientnet"
else:
    # model = eda.models.densenet121(weights=None)
    # model.classifier = eda.nn.Linear(model.classifier.in_features, len(LABELS))
    model = eda.models.efficientnet_b3(weights=None)
    model.load_state_dict(eda.torch.load(args.model_version))
    # architecuter = "densenet"
    architecuter = "efficientnet"
    # this method was created in a desperation but not necessary correct
    # model = densenet3.create_densenet3(len(LABELS), args.model_version, use_cache=True)

def last_layers_of_densenet(name):
    return 'classifier' in name or 'norm5' in name or 'denseblock4' in name # or re.match('.*denseblock4.denselayer1[1-9].*', name)

if args.freeze_begin:
    print("Freeze begins")
    # try the articles idea of freezing main layers first
    for param in model.parameters():
            param.requires_grad = False
    if architecuter == "resnet":
        for name, param in model.named_parameters():
            if 'fc' in name or 'layer4' in name or 'layer3' in name:
                param.requires_grad = True
    elif architecuter == "densenet":
        for name, param in model.named_parameters():
            if last_layers_of_densenet(name):
                param.requires_grad = True
    else:
        raise "not implemented freeze_begin arch"
else:
    print("long execution")


import torch.utils.checkpoint as checkpoint
def faster_densenet_forward(self, x):
    return checkpoint.checkpoint_sequential(self.features, 4, x)
if args.randomize_end:
    print("random weights")
    # for learning acceleration
    if architecuter == "densenet":
        for name, module in model.named_modules():
            if last_layers_of_densenet(name):
                randomize_weights_densenet(module)
        # hidden_layer = 64
        # model.classifier = nn.Sequential(
        #         nn.Linear(model.classifier.in_features, hidden_layer),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.1),
        #         nn.Linear(hidden_layer, len(LABELS))
        #     )
    else:
        raise "non implememnted randomize_end arch"

transform = eda.transforms.Compose([
        eda.transforms.ToTensor(),
        eda.transforms.Normalize(mean=[0.479, 0.455, 0.461], std=[0.109, 0.104, 0.105]),
        # eda.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

import albumentations as alb

augmentation = alb.Compose([
            alb.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-170, 170),
                p=1),
            alb.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0,
                p=0.9
            )
            ])

Path('/home/kpetrenko/work/models/').mkdir()

# torch._C._set_tracing_state(None)
train_loader, val_loader = eda.create_cpu_friendly_dataloaders(data, transform, augemntation_pipeline=augmentation, consider_small=1000, consider_mild=3000)
# train_loader, val_loader = eda.create_cpu_friendly_dataloaders(data, transform)
progress_loss_acc = eda.cpu_friendly_train(device, model, filter(lambda p: p.requires_grad, model.parameters()), train_loader, val_loader, epochs=15, background_run=True, checkpoint_frequency=3)
