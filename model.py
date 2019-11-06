import torchvision.models as models
import torch.nn as nn

def initialize_model():
    net = models.mobilenet_v2()
    n_features = 160

    # Replace first layer to use 9 channels instead of 3
    features_children = list(net.features.children())
    features_children[0] = models.mobilenet.ConvBNReLU(9, 32, kernel_size=3, stride=(2,2))
    features_children[-1] = models.mobilenet.ConvBNReLU(320, n_features, kernel_size=1, stride=(1,1))
    net.features = nn.Sequential(*features_children)

    # Replace classifier to predict low/med/high night light intensity
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=n_features, out_features=3, bias=True)
    )
    return net