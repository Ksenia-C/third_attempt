import datasets
import pandas as pd
import pickle
from datasets import Dataset
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Lambda

dataset = datasets.load_dataset('flwrlabs/fed-isic2019')
dataset = pd.DataFrame(dataset['train']).sample(10000)

pytorch_transforms = Compose([Resize((128, 128))])

# with open('dataset.pkl', 'rb') as f:
#     dataset = pickle.load(f)
dataset['image'] = dataset['image'].apply(pytorch_transforms)
# dataset = Dataset.from_dict(dataset.reset_index(drop=True).to_dict(orient='list'))

with open("dataset_isic.pkl", 'wb') as f:
    # pickle.dump(dataset.to_pandas(), f)
    pickle.dump(dataset, f)
    
