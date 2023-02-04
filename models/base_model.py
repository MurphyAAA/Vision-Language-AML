import torch
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        # x shape: Batch_size * Channel * H * W #  32*3*224*224
        x = self.resnet18.conv1(x) # 32*64*112*112
        x = self.resnet18.bn1(x) # 32*64*112*112
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x) # 32*64*56*56
        x = self.resnet18.layer1(x) # 32*64*56*56
        x = self.resnet18.layer2(x) # 32*128*28*28
        x = self.resnet18.layer3(x) # 32*256*14*14
        x = self.resnet18.layer4(x) # 32*512*7*7
        x = self.resnet18.avgpool(x) # 32*512*1*1
        x = x.squeeze()

        if len(x.size()) < 2:
            x = x.unsqueeze(0)

        return x

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        # model中包含了 特征提取器和分类器
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), # encoder就是得到一个vector，classifer就是把这个vector经过全连接得到n个类
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7) # 全连接

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        # domain_encoder, category_encoder都是 Disentangler()，两个encoder分开写了
        # 变成只有domain信息的vector
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # encoder就是得到一个vector，classifer就是把这个vector经过全连接得到n个类
            nn.ReLU()
        )

        # 变成只有category信息的vector
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # encoder就是得到一个vector，classifer就是把这个vector经过全连接得到n个类
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512,4)
        self.category_classifier = nn.Linear(512,7)

        self.reconstructor = nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU()
        )

    def forward(self, x): # xd包含source+target domain的图

        x = self.feature_extractor(x) # 没有 category label的也正常参加处理，只是计算loss，更新梯度时排除掉
        fcs = self.category_encoder(x)
        fds = self.domain_encoder(x)

        # need to return
        fG_hat = torch.cat((fds,fcs),dim=1)
        fG_hat = self.reconstructor(fG_hat)

        Cfcs = self.category_classifier(fcs) # 经过classifier之后再传出去，nn自带的CrossEntropy本身包括了logSoftmax的计算
        DCfcs = self.domain_classifier(fcs) #??????????? 这个要放外面算？？？，要冻住DC，反向传播不能更新domain_classifier

        DCfds = self.domain_classifier(fds)
        Cfds = self.category_classifier(fds)

        return x, fG_hat, Cfcs, DCfcs, DCfds, Cfds


class CLIPDisentangleModel(nn.Module): # 就多返回一个fd
    def __init__(self):
        super(CLIPDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder= nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # encoder就是得到一个vector，classifer就是把这个vector经过全连接得到n个类
            nn.ReLU()
        )
        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),  # encoder就是得到一个vector，classifer就是把这个vector经过全连接得到n个类
            nn.ReLU()
        )
        self.category_classifier = nn.Linear(512,7)
        self.domain_classifier = nn.Linear(512,4)

        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
        )

    def forward(self, x):

        x = self.feature_extractor(x)
        fcs = self.category_encoder(x)
        fds = self.domain_encoder(x)

        # need to return
        fG_hat = torch.cat((fds, fcs), dim=1)
        fG_hat = self.reconstructor(fG_hat)

        Cfcs = self.category_classifier(fcs)
        DCfcs = self.domain_classifier(fcs) # 使category encoder排除掉domain的信息

        DCfds = self.domain_classifier(fds)
        Cfds = self.category_classifier(fds)


        return x, fG_hat, Cfcs, DCfcs, DCfds, Cfds, fds