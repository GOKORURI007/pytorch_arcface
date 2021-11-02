# pytorch_arcface

Pytorch implementation of Arcface Loss / AAM Softmax Loss

# How to use

There are three implementations of Arcface Loss / AAM Softmax Loss in `class ArcFace` in arcface.py. Just choose one of these and change its' name from `forward1/2/3(...)` to `forward(...)` to use it as a normal 'torch.nn.Module'. speed_test.py is a script to test the inference speed of different implementations and comfirm that these method are equivalent.

Example:
```
from torch import nn
from arcface import ArcFace
class FeatureExtractor(nn.Module):
    def __init__(self, embd_size, ...):
        self.conv1 ...
        self.conv2 ...
        ...
        self.fc = nn.Linear(input_dim, embd_size)
    def forward(self, inpt):
        x = self.conv1(inpt)
        x = self.conv2(x)
        ...
        x = self.fc(x)
        return x

model = FeatureExtractor(...)
loss = ArcFace(...)

for epoch in range(max_epoch):
    for inpt, label in your_train_dataloader:
        embeds = model(inpt)
        loss = loss(embeds, label)
    
    # your optimize process
    ...

```

If you want to get the probability, modify the `forward(...)` like

```
def forward(self, embed, label):
    ...
    output *= self.scale
    loss = self.ce(...)
    return loss, output
```


# References
https://github.com/deepinsight/insightface

https://github.com/ronghuaiyang/arcface-pytorch

# Citation

If you find ArcFace Loss useful in your research, please consider to cite the following paper:

```
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```
