import eda_helpers as eda
import psutil
import torch
import json

DATASET_PATH = 'flwrlabs/fed-isic2019'
LABELS = list(map(str, range(8)))
GROUP_COL = 'center'

device = torch.device('cpu')
torch.set_num_threads(min(20, psutil.cpu_count()))
data = eda.download_dataset(DATASET_PATH, 'train')
data['image'] = data['image'].map(lambda x: eda.resize_and_pad(x, (224, 224)))

# Usage
model = eda.models.densenet121(weights=None)
model.classifier = eda.nn.Linear(model.classifier.in_features, len(LABELS))

transform = eda.transforms.Compose([
        eda.transforms.ToTensor(),
        eda.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_loader, val_loader = eda.create_cpu_friendly_dataloaders(data, transform)
progress_loss_acc = eda.cpu_friendly_train(device, model, train_loader, val_loader, epochs=20, background_run=True)

with open("/home/kpetrenko/work/models/skin_progress", 'w') as f:
    # training_acc
    json.dump(progress_loss_acc[0][0], f)
    f.write('\n')

    # validation_acc
    json.dump(progress_loss_acc[0][1], f)
    f.write('\n')

    # training_loss
    json.dump(progress_loss_acc[1][0], f)
    f.write('\n')

    # validation_loss
    json.dump(progress_loss_acc[1][1], f)
    f.write('\n')
