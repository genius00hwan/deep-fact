import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image

import warnings

warnings.filterwarnings('ignore')

from model import Xception
from f3net import F3Net


def return_acc(file_path):
    image = Image.open(file_path).convert('RGB')

    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    image = test_transform(image)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CPU 환경만을 가정
    model = F3Net()
    # model.load_state_dict(torch.load('./F3Net.pth.tar', map_location=torch.device('cpu'))['model_state_dict'])

    output = model(image.unsqueeze(0))  # 한 개의 이미지를 가정
    _, preds = torch.max(output, 1)

    return preds[0]
