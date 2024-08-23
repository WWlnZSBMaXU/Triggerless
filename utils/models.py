import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
from utils.datasets import feature_sizes
import torch.nn.functional as F
 
 
def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Flatten(nn.Module):
    '''Flatten the input'''
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv4_passive(nn.Module):
    '''
    Conv4 passive model for CIFAR-10 and CINIC-10.
    '''
    def __init__(self, num_passive):
        super(Conv4_passive, self).__init__()
        self.num_passive = num_passive

        if num_passive not in [1, 2, 4, 8]:
            raise ValueError("The number of passive parties must be 1, 2, 4 or 8.")
        
        self.embeddings = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64)
        )        
        print("Passive Model", self.embeddings)
       
    def forward(self, x):
        emb = self.embeddings(x)
        return emb
    

class Conv4_active(nn.Module):
    '''
    Conv4 active model for CIFAR-10 and CINIC-10.
    '''
    def __init__(self, num_classes, num_passive, division):
        super(Conv4_active, self).__init__()
        self.num_passive = num_passive

        if division == 'imbalanced' and num_passive == 4:
            input_size = 8 * 7 * 64
        else:
            input_size = 8 * 8 * 64

        self.prediction = nn.Sequential(
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        print("Active Model", self.prediction)
        
    def forward(self, x):
        logit = self.prediction(x)
        return logit

    
class Conv4(nn.Module):
    '''
    Conv4 model for CIFAR-10 and CINIC-10.
    '''
    def __init__(self, num_classes, num_passive, division):
        super(Conv4, self).__init__()
        self.num_passive = num_passive

        self.passive = []
        for _ in range(num_passive):
            self.passive.append(Conv4_passive(num_passive))

        self.active = Conv4_active(num_classes, num_passive, division)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))

        agg_emb = self._aggregate(emb)
        logit = self.active(agg_emb)

        pred = self.softmax(logit)

        return emb, logit, pred
    
    def _aggregate(self, x):
        '''
        Aggregate the embeddings from passive parties.
        '''
        return torch.cat(x, dim=3)

    
class Flatten_input(nn.Module):
    '''Flatten the input'''
    def __init__(self):
        super(Flatten_input, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)
    

class FC1_passive(nn.Module):
    '''
    FC1 passive model for MNIST and FashionMNIST.
    28 * 28 * 1
    '''
    def __init__(self, num_passive, linear_size):
        super(FC1_passive, self).__init__()

        if num_passive not in [1, 2, 4, 7]:
            raise ValueError("The number of passive parties must be 1, 2, 4 or 7.")

        emb_size = int(28 * (28 / num_passive))
        self.embeddings = nn.Sequential(
            Flatten_input(),
            nn.Linear(linear_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(inplace=False)
        )

        print("Passive Model", self.embeddings)
       
    def forward(self, x):
        emb = self.embeddings(x)
        return emb

    
class FC1_active(nn.Module):
    '''
    FC1 active model for MNIST and FashionMNIST.
    28 * 28 * 1
    '''
    def __init__(self):
        super(FC1_active, self).__init__()

        self.prediction = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(28 * 28, 10)
        )
        print("Active Model", self.prediction)
        
    def forward(self, x):
        logit = self.prediction(x)
        return logit


class FC1(nn.Module):
    '''
    FC1 model for MNIST and FashionMNIST.
    28 * 28 * 1
    '''
    def __init__(self, num_passive, division):
        super(FC1, self).__init__()
        self.num_passive = num_passive

        linear_size_list = []
        if division in ['vertical', 'random']:
            linear_size_list = [int(28 * (28 / num_passive))] * num_passive
        elif division == 'imbalanced':
            if num_passive == 1:
                linear_size_list = [28 * 28]
            elif num_passive == 2:
                linear_size_list.append(28 * 20)
                linear_size_list.append(28 * 8)
            elif num_passive == 4:
                linear_size_list.append(28 * 12)
                linear_size_list.append(28 * 6)
                linear_size_list.append(28 * 3)
                linear_size_list.append(28 * 7)

        self.passive = []
        for i in range(num_passive):
            self.passive.append(FC1_passive(num_passive, linear_size_list[i]))

        self.active = FC1_active()
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))

        agg_emb = self._aggregate(emb)
        logit = self.active(agg_emb)

        pred = self.softmax(logit)

        return emb, logit, pred
    
    def _aggregate(self, x):
        '''
        Aggregate the embeddings from passive parties.
        '''
        # Note: x is a list of tensors.
        return torch.cat(x, dim=1)


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False
    )


class ResidualBlock(nn.Module):
    '''
    Residual Block for ResNet.
    '''
    def __init__(self, inchannel, outchannel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.block_conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=False),
            nn.Conv2d(outchannel,outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
 
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self,x):
        out1 = self.block_conv(x)
        out2 = self.shortcut(x)+out1
        out2 = F.relu(out2)
        return out2


class ResNet_passive(nn.Module):
    '''
    ResNet passive model for CIFAR-100.
    '''
    def __init__(self, block, layers, num_passive):
        super(ResNet_passive, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            # (n-f+2*p)/s+1,n=28,n=32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), #64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0) #64
        )
 
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1) #64
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2) #32
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2) #16
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2) #8
        
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet_active(nn.Module):
    '''
    ResNet active model for CIFAR-100.
    '''
    def __init__(self, num_classes):
        super(ResNet_active, self).__init__()
        self.linear = nn.Linear(512*1*1,num_classes) 
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ResNet(nn.Module):
    '''
    ResNet model for CIFAR-100.
    '''
    def __init__(self, block, layers, num_classes, num_passive, **kwargs):
        super(ResNet, self).__init__()
        self.num_passive = num_passive

        self.passive = []
        for _ in range(num_passive):
            self.passive.append(ResNet_passive(block, layers, num_passive))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.active = ResNet_active(num_classes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))
        agg_emb = torch.cat(emb, dim=3)
        agg_emb = self.avgpool(agg_emb)
        agg_emb = agg_emb.view(agg_emb.size(0), -1)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)

        return emb, logit, pred
    
    def _aggregate(self, x):
        agg_emb = torch.cat(x, dim=1)
        agg_emb = agg_emb.view(agg_emb.size(0), -1)
        return agg_emb


class DeepFM_passive(nn.Module):
    '''
    DeepFM passive model for Criteo.
    '''
    def __init__(self, feature_sizes, emb_size, hidden_size, dropout):
        super(DeepFM_passive, self).__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(fz, emb_size) for fz in feature_sizes])
        self.DNN = nn.Sequential(
            nn.Linear(len(feature_sizes) * emb_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout)
        )
        print("Passive Model", self.embeddings, self.DNN)

    def forward(self, x):
        # FM part
        # - Xi: A tensor of input's index, shape of (N, field_size, 1)
        # - Xv: A tensor of input's value, shape of (N, field_size, 1)
        Xi = (
            torch.cat([torch.zeros_like(x[:, :2]), x[:, 2:]], dim=-1)
            .unsqueeze(-1)
            .int()
        )
        Xv = torch.cat(
            [x[:, :2], torch.ones_like(x[:, 2:])], dim=-1
        ).float()

        emb_fm = [
            (torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t()
            for i, emb in enumerate(self.embeddings)
        ]

        # Deep part
        emb_deep = torch.cat(emb_fm, dim=1)
        emb_deep = self.DNN(emb_deep)

        return emb_deep


class DeepFM_active(nn.Module):
    '''
    DeepFM active model for Criteo.
    '''
    def __init__(self, hidden_size, dropout, num_classes, num_passive):
        super(DeepFM_active, self).__init__()
        if num_passive == 1:
            self.prediction = nn.Sequential(
                nn.Linear(hidden_size, num_classes)
            )
        elif num_passive == 3:
            self.prediction = nn.Sequential(
                nn.Linear(hidden_size * num_passive, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            raise ValueError("The number of passive parties must be 1 or 3.")
        print("Active Model", self.prediction)

    def forward(self, x):
        logit = self.prediction(x)
        return logit


class DeepFM(nn.Module):
    '''
    DeepFM model for Criteo.
    '''
    def __init__(self, feature_sizes, emb_size, hidden_size, dropout, num_classes, num_passive, **kwargs):
        super(DeepFM, self).__init__()
        self.num_passive = num_passive

        feature_size_list = []
        feature_stride = int(len(feature_sizes) / num_passive)
        for i in range(num_passive):
            feature_size_list.append(feature_sizes[i*feature_stride: (i+1)*feature_stride])

        self.passive = []
        for i in range(num_passive):
            self.passive.append(DeepFM_passive(feature_size_list[i], emb_size, hidden_size, dropout))

        self.active = DeepFM_active(hidden_size, dropout, num_classes, num_passive)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = list(x)
        emb = []
        for i in range(self.num_passive):
            emb.append(self.passive[i](x[i]))
        
        agg_emb = self._aggregate(emb)
        logit = self.active(agg_emb)
        pred = self.softmax(logit)

        return emb, logit, pred
    
    def _aggregate(self, x):
        return torch.cat(x, dim=1)


entire = {
    'mnist': FC1,
    'fashionmnist': FC1,
    'cifar10': partial(Conv4, num_classes=10),
    'cifar100': partial(ResNet, block=ResidualBlock, layers=[2, 2, 2, 2], num_classes=100),
    'criteo': partial(DeepFM, feature_sizes=feature_sizes, emb_size=4, hidden_size=32, dropout=0.5, num_classes=2),
    'cinic10': partial(Conv4, num_classes=10)
}