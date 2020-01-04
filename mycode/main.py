from utils.myloader import MyLoader
from utils.imageshow import func_showImage, func_rearrangeRGB
from models import FCNs
from config import *

from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt


transforms = T.Compose([
    T.ToTensor(),
])

trainset = MyLoader(path_src="../data/camseq01", path_label="../data/mask", transforms=transforms)
trainloader = DataLoader(trainset, num_workers=1, batch_size=1, shuffle=True)

train_iter = iter(trainloader)


net = FCNs().cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):
    for idx, (img, label) in enumerate(train_iter):
        img = img.cuda()
        label = label.cuda()

        out = net(img)
        optimizer.zero_grad()
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()

        if idx % 50 == 0 and idx != 0:
            l = out[0][6].detach().cpu().numpy()
            plt.imshow(l)
            plt.show()

            print("Loss: %.6f" % loss)

    train_iter = iter(trainloader)

torch.save(net.state_dict(), PATH)