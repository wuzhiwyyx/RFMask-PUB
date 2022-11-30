'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:13:24
 # @ Description: Branch definition, containing Encoder module.
 '''


import torch.nn as nn


class RFPose2DEncoder(nn.Module):
    """RFPose2DEncoder definition."""

    def __init__(self, in_channels=2, out_channels=64, 
                    mid_channels=64, layer_num=10, seq_len=12) -> None:
        """
        RFPose2DEncoder constructor.

        Args:
            in_channels (int, optional): Input data channel. Defaults to 2.
            out_channels (int, optional): Output feaature channel. Defaults to 64.
            mid_channels (int, optional): Module width of hidden layers. Defaults to 64.
            layer_num (int, optional): Total number of layers. Defaults to 10.
            seq_len (int, optional): Input sequence length. Defaults to 12.
        """
        super().__init__()
        layers = []
        for i in range(layer_num // 2):
            if i == 0:
                if seq_len < 9:
                    layers.extend(self.conv3d_layer(in_channels, mid_channels, k1=(3, 5, 5), p1=(1, 2, 2)))
                else:
                    layers.extend(self.conv3d_layer(in_channels, mid_channels))
            elif i == layer_num // 2 - 1:
                layers.extend(self.conv3d_layer(mid_channels, out_channels, s1=(2, 2, 2)))
            else:
                layers.extend(self.conv3d_layer(mid_channels, mid_channels))
        self.conv = nn.Sequential(*layers)

    def conv3d_layer(self, in_channels, out_channels, 
                     k1=(9, 5, 5), s1=(1, 2, 2), 
                     k2=(9, 5, 5), s2=(1, 1, 1), 
                     p1=(4, 2, 2), p2=(4, 2, 2)):
        res = [
            nn.Conv3d(in_channels , out_channels, k1, s1, p1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, k2, s2, p2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        return res
    
    def forward(self, x):
        return self.conv(x)

class RFPose2DDecoder(nn.Module):
    """RFPose2DDecoder definition."""

    def __init__(self, in_channels=32, out_channels=1) -> None:
        """RFPose2DDecoder constructor.

        Args:
            in_channels (int, optional): Input feature channel. Defaults to 32.
            out_channels (int, optional): Output result channel. Defaults to 1.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 64, (3, 6, 6), (1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(64, 32, (3, 6, 6), (1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),
            nn.ConvTranspose3d(32, 16, (3, 6, 6), (1, 2, 2), padding=(1, 2, 2)),
            nn.PReLU(),            
            nn.ConvTranspose3d(16, out_channels, (3, 6, 6), (1, 4, 4), padding=(1, 4, 4)),
#             nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x