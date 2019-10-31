import torchvision.models as models
import torch.nn as nn

def initialize_model():
    net = models.mobilenet_v2()

    # Replace first layer to use 9 channels instead of 3
    features_children = list(net.features.children())
    features_children[0] = models.mobilenet.ConvBNReLU(9, 32, kernel_size=3, stride=(2,2))
    net.features = nn.Sequential(*features_children)

    # Replace classifier to predict low/med/high night light intensity
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=1280, out_features=3, bias=True)
    )
    return net