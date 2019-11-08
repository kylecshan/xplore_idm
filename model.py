import torchvision.models as models
import torch.nn as nn

def initialize_model(n_features=100):
    net = models.mobilenet_v2()

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
    
    net.n_features = n_features
    return net

def initialize_model2(n_features=100):
    net = models.vgg11_bn()

    # Replace first layer to use 9 channels instead of 3
    net.features[0] = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    net.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    # Replace classifier to predict low/med/high night light intensity
    net.classifier = nn.Sequential(
#         nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=512, out_features=3, bias=True)
    )
    
    net.n_features = 512
    return net

def initialize_model3(n_features=100):
    net = models.resnet50()

    # Replace first layer to use 9 channels instead of 3
    net.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
#     net.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    # Replace classifier to predict low/med/high night light intensity
    net.fc = nn.Sequential(
#         nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=2048, out_features=3, bias=True)
    )
    
    net.n_features = 2048
    return net