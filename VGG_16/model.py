import torch.nn as nn
import matplotlib.pyplot as plt
import os
import imageio
import glob


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def create_gif(image_path_pattern, gif_path, duration=0.5):
    images = []
    for filename in sorted(glob.glob(image_path_pattern)):
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_path, images, duration=duration)



class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[self.vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        self.conv_outputs = []  # Reset at each forward pass
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.conv_outputs.append(x)  # Store conv layer output
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
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
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
