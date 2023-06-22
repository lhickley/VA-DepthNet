#Just an easy exposed helper script to take an input and actally run the model and get a prediction

import torch
import argparse
import numpy as np
import os
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt

from dataloaders.dataloader import NewDataLoader
from networks.vadepthnet import VADepthNet
from torchvision import transforms

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def load_image(image_path):
    # Load image, then convert it to RGB and normalize it to [0, 1]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

parser = argparse.ArgumentParser(description='VADepthNet PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--prior_mean',                type=float, help='prior mean of depth', default=1.54)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='/Users/lewishickley/BBK/MscThesis/models/initialSet/VA-DepthNet/models/vadepthnet_nyu.pth')
parser.add_argument('--image_path',                type=str,   help='path to the image', default='/Users/lewishickley/Downloads/InitialTestImage.jpg') #TODO change this default and the handling around it.


args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

model = VADepthNet(max_depth=args.max_depth, 
                       prior_mean=args.prior_mean, 
                       img_size=(args.input_height, args.input_width))

model.train()

num_params = sum([np.prod(p.size()) for p in model.parameters()])
print("== Total number of parameters: {}".format(num_params))

num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
print("== Total number of learning parameters: {}".format(num_params_update))

model = torch.nn.DataParallel(model)
#model.cuda() #Turn this back on one day.

print("== Model Initialized")

if os.path.isfile(args.checkpoint_path):
    print("== Loading checkpoint '{}'".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
    del checkpoint
else:
    print("== No checkpoint found at '{}'".format(args.checkpoint_path))

cudnn.benchmark = True #TODO Consider turning this off when we have the various models all together, might slow us down depending on how they are blended.

model.eval()

img = load_image(args.image_path)
img = transform(img)
img = img.unsqueeze(0).float()

with torch.no_grad():  # Do not calculate gradients
        output = model(img)

# The output is a depth map, it's up to us how to process it
# For simplicity, let's convert it to numpy and squeeze unnecessary dimensions
output = output.cpu().numpy().squeeze()

#Return a plot of the data so we can visualise how it is doing.
#TODO Write a function to save this image.
plt.imshow(output, cmap='inferno')
plt.colorbar()
plt.show()