import torch
import torch.nn as nn
import torch.nn.functional as F
from RandomHingeForest import RandomHingeForestFusedLinear
from HingeTree import expand

class Net(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()

        number_of_features=100
        number_of_trees=100
        depth=7

        self.extra_outputs=[16,16,4] # For MONAI

        #self.dropout = nn.Dropout3d(p=0.2, inplace=True)
        self.dropout = lambda x : x

        #self.conv11 = nn.Conv3d(in_channels, 40, 5, stride=[1,4,4], padding=2, bias=False)
        self.conv11 = nn.Conv3d(in_channels, 40, 5, stride=[4,4,1], padding=2, bias=False)
        #self.conv11 = nn.Conv3d(in_channels, 40, 3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm3d(40, affine=False)
        self.conv12 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm3d(40, affine=False)
        self.conv13 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm3d(40, affine=False)

        self.pool = nn.MaxPool3d(2,2)

        self.conv21 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn21 = nn.BatchNorm3d(40, affine=False)
        self.conv22 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn22 = nn.BatchNorm3d(40, affine=False)
        self.conv23 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn23 = nn.BatchNorm3d(40, affine=False)

        self.conv31 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn31 = nn.BatchNorm3d(40, affine=False)
        self.conv32 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn32 = nn.BatchNorm3d(40, affine=False)
        self.conv33 = nn.Conv3d(40, 40, 3, padding=1, bias=False)
        self.bn33 = nn.BatchNorm3d(40, affine=False)

        self.features = nn.Conv3d(40, number_of_features, 1, bias=False)
        self.forestbn = nn.BatchNorm3d(number_of_features, affine=False)

        #self.forest = RandomHingeForestFusedLinear(number_of_features, number_of_trees, out_channels, depth=depth, extra_outputs=[4,16,16])
        self.forest = RandomHingeForestFusedLinear(number_of_features, number_of_trees, out_channels, depth=depth, extra_outputs=self.extra_outputs)
        #self.forest = RandomHingeForestFusedLinear(number_of_features, number_of_trees, out_channels, depth=depth, extra_outputs=[8,8,8])

    def calculate_features(self, x):
        x = F.relu(self.bn11(self.dropout(self.conv11(x))))
        x = F.relu(self.bn12(self.dropout(self.conv12(x))))
        x = F.relu(self.bn13(self.dropout(self.conv13(x))))
        x = self.pool(x)

        x = F.relu(self.bn21(self.dropout(self.conv21(x))))
        x = F.relu(self.bn22(self.dropout(self.conv22(x))))
        x = F.relu(self.bn23(self.dropout(self.conv23(x))))
        x = self.pool(x)
        
        x = F.relu(self.bn31(self.dropout(self.conv31(x))))
        x = F.relu(self.bn32(self.dropout(self.conv32(x))))
        x = F.relu(self.bn33(self.dropout(self.conv33(x))))
        #x = self.pool(x)

        x = self.forestbn(self.features(x))

        return x

    def forward(self, x):
        x = self.calculate_features(x)
        x = self.forest(x)

        return x

    def leafmap(self, x):
        origShape = x.shape

        x = self.calculate_features(x)

        #x = self.forestbn1(x)
        #maps = x

        x = self.forest.leafmap(x)

        #print(maps[:,23,:,:][x[:,0,:,:] == 22].max())

        x = F.interpolate(x, size=origShape[2:], mode="nearest")

        return x

if __name__ == "__main__":
    device="cuda:0"

    net = Net().to(device)

    fileName="/data/AMPrj/AllImages/Models/snapshots_everything_contrastive/epoch_24.pt"

    params = torch.load(fileName, map_location=device)
    newParams = dict()

    #for key, _ in net.named_parameters():
    #    if key in params:
    #        newParams[key] = params[key]

    for key in net.state_dict():
        if key in params:
            newParams[key] = params[key]

    del params

    net.load_state_dict(newParams)

    x = torch.randn([8, 4, 80, 272, 176]).to(device)

    print(x.shape)

    x = net(x)

    x = x.cpu()
    x = expand(x)

    print(x.shape)

