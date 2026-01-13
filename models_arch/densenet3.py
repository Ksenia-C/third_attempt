# Code below was likely half generated half added by my hands (the cache part)
# (I remember weakly - it was late at night when I developed it)
# It matters not so much since the code is incorrect by idea
# But it made the training 15-times faster
import torch
import torch.nn as nn
from torchvision import models

class TensorCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def _get_key(self, tensor):
        """Create a proper cache key from tensor"""
        return hash(tensor.cpu().numpy().tobytes())

    def __contains__(self, tensor):
        key = self._get_key(tensor)
        if key not in self.cache:
            return False 
        for (cached_tensor, _) in self.cache[key]:
            if torch.equal(tensor, cached_tensor):
                return True
        return False 
    
    def __getitem__(self, tensor):
        key = self._get_key(tensor)
        if key in self.cache:
            for (cached_tensor, value) in self.cache[key]:
                if torch.equal(tensor, cached_tensor):
                    return value
        raise KeyError("Tensor not in cache")
    
    def __setitem__(self, tensor, value):
        key = self._get_key(tensor)        
        self.cache[key] = self.cache.get(key, []) + [(tensor, value)]
    
    def clear(self):
        self.cache.clear()

    def len(self):
        return len(self.cache)


def create_densenet3(num_classes, weights_path, use_cache=False):
    """
    Create DenseNet121 truncated after denseblock3/transition3
    """
    # Load full model
    full_model = models.densenet121(weights=None)
    full_model.classifier = nn.Linear(full_model.classifier.in_features, num_classes)
    full_model.load_state_dict(torch.load(weights_path))
    full_model = full_model.to(memory_format=torch.channels_last)
    
    # Extract everything up to and including transition3
    # DenseNet121 features structure:
    # [conv0, norm0, relu0, pool0, denseblock1, transition1, 
    #  denseblock2, transition2, denseblock3, transition3, 
    #  denseblock4, norm5]
    
    # Create a sequential model with layers up to transition3
    truncated_features = nn.Sequential(*list(full_model.features.children())[:10])
    # This includes: conv0, norm0, relu0, pool0, denseblock1, transition1, 
    #                denseblock2, transition2, denseblock3, transition3
    
    class TruncatedDenseNet121(nn.Module):
        def __init__(self, features, num_classes, use_feature_cache=False):
            super().__init__()
            self.use_feature_cache = use_feature_cache
            self.sized_cache = TensorCache()

            self.features = features
            
            # Calculate the output features after transition3
            # For DenseNet121, transition3 output is 512 channels
            self.num_features = 512
            
            # Global pooling and classifier
            self.relu = nn.ReLU(inplace=True)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(self.num_features, num_classes)

            
        def forward(self, x):
            def clean_run():
                features = self.features(x)
                if self.use_feature_cache and self.sized_cache.len() < 1000:
                    self.sized_cache[x] = features
                return features
            if self.use_feature_cache:
                if x in self.sized_cache:
                    features = self.sized_cache[x]
                else:
                    features = clean_run()
            else:
                features = clean_run()
            out = self.relu(features)
            out = self.global_pool(features)
            out = out.view(features.size(0), -1)
            out = self.classifier(out)
            return out
    
    full_model = TruncatedDenseNet121(truncated_features, num_classes, use_cache)
    full_model = torch.compile(full_model, backend="inductor")
    full_model = full_model.to(memory_format=torch.channels_last)
    return full_model
