import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim=20, only_pred=False):
        super().__init__()
        self.only_pred = only_pred
        self.encoder = nn.Sequential(nn.Linear(input_dim, 8),
                                     nn.LeakyReLU(),
                                     nn.Linear(8, 8),
                                     nn.ReLU())
        self.regression_layer = nn.Linear(8, 1)
        self.decoder = nn.Linear(8, 20)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        pred = self.regression_layer(features)
        if self.only_pred:
            return pred
        else:
            return pred, features

    def forward_decoder(self, x):
        features = self.encoder(x)
        rec = self.decoder(features)
        return rec, features



class MLP_unc(nn.Module):

    def __init__(self, m=100, dim_x=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(m + dim_x, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 100),
                                     nn.ReLU())
        self.regression_layer = nn.Linear(100, 1)
        self.regression_unc = nn.Linear(100, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        pred = self.regression_layer(features)
        var = self.regression_unc(features)
        return pred, var



class MLP_classification(nn.Module):

    def __init__(self, m=100, dim_x=1, bins=100):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(m + dim_x, 100),
                                     nn.ReLU(),
                                     nn.Linear(100, 100),
                                     nn.ReLU())
        self.classification_layer = nn.Linear(100, bins)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        pred = self.classification_layer(features)

        return pred, features