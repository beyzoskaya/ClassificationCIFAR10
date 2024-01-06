import torch.nn as nn
import matplotlib.pyplot as plt
import os
import imageio
import glob
from PIL import Image
from natsort import natsorted
import torch

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # this and below candidate
    'VGG22': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 'M', 512, 'M'],
    'VGG26': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 'M', 512, 512, 'M']
}


def create_gif(image_path_pattern, gif_path, duration=0.5):
    images = []
    for filename in sorted(glob.glob(image_path_pattern)):
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_path, images, duration=duration)

def create_plot(path):
    imgs = []
    for layer in natsorted(os.listdir(path)):
        print(layer)
        channels = os.path.join(path, layer)
        imgs.append(Image.open(os.path.join(channels, sorted(os.listdir(channels))[0])))
    plt.figure(figsize=(20, 10))  # Adjust the size as needed
    for idx, img in enumerate(imgs):
        ax = plt.subplot(1, len(imgs), idx + 1)  # Creates a subplot for each image
        ax.imshow(img)
        ax.set_title(f"Layer_{idx}")  # The layer name is set as the title of each subplot
        ax.axis('off')  # Turn off axis

    plt.tight_layout()
    plt.savefig("./plot_convs.png")
        
class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, init_size=(32, 32)):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.num_classes = num_classes

        # Dynamically adjust the classifier
        with torch.no_grad():
            self._initialize_classifier(init_size)

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_classifier(self, init_size):
        # Forward a dummy input through the feature layers to determine output size
        dummy_input = torch.zeros(1, 3, *init_size)
        output_size = self.features(dummy_input).view(-1).shape[0]
        self.classifier = nn.Linear(output_size, self.num_classes)
    
    def save_feature_maps(self, output_dir='feature_maps'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for layer_num, feature_map in enumerate(self.conv_outputs):
            layer_dir = os.path.join(output_dir, f'layer_{layer_num}')
            os.makedirs(layer_dir, exist_ok=True)

            num_channels = feature_map.size(1)
            for channel in range(num_channels):
                # Normalize the feature map for visualization
                fmap_normalized = feature_map[0, channel].detach().cpu().numpy()
                fmap_normalized = (fmap_normalized - fmap_normalized.min()) / (fmap_normalized.max() - fmap_normalized.min())

                plt.imsave(os.path.join(layer_dir, f'channel_{channel}.png'), fmap_normalized, cmap='viridis')
        create_gif('./feature_maps/layer_0/*.png', 'layer_0.gif', duration=0.5)
        create_plot('./feature_maps')
