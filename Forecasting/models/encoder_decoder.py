import torch
import torch.nn as nn
from Forecasting.config import cfg


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Encoder_Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding, verbose=False):
        super().__init__()
        self.batch = b_h_w[0]
        self.height = b_h_w[1]
        self.width = b_h_w[2]
        self.verbose = verbose

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=input_channel, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def encode(self, x):
        if self.verbose:
            print(x.shape) # torch.Size([16, 10, 6, 81, 91])
        x = torch.nn.functional.pad(x, (2, 3, 7, 8))
        if self.verbose:
            print(x.shape) # torch.Size([16, 10, 6, 96, 96])
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        if self.verbose:
            print(x.shape) # torch.Size([16, 60, 96, 96])
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        if self.verbose:
            print(x5.shape) #torch.Size([16, 512, 12, 12])
            print('\n----------------------------------')

        return x5

    def decode(self, y):
        # y1 = self._Up1(y)
        # if self.verbose:
        #     print(y1.shape)  # torch.Size([16, 256, 24, 24])
        #
        # y1a = self._Att1(y1, y1)
        # if self.verbose:
        #     print(y1a.shape) #
        #
        # y2 = self._Up2(y1a)
        # if self.verbose:
        #     print(y2.shape) # torch.Size([16, 128, 48, 48])
        #
        # y2a = self._Att2(y1a, y2)
        # if self.verbose:
        #     print(y2a.shape) #
        #
        # y3 = self._Up3(y2a)
        # if self.verbose:
        #     print(y3.shape) # torch.Size([16, 64, 96, 96])
        # y4 = self._Conv_1x1(y3)
        # if self.verbose:
        #     print(y4.shape) # torch.Size([16, 60, 94, 94])

        # decoding + concat path
        d5 = self.Up5(y)
        x4 = self.Att5(g=d5, x=d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        output = torch.reshape(d1, (-1, cfg.out_len, cfg.features_amount, d1.shape[2], d1.shape[3]))
        if self.verbose:
            print(output.shape) # torch.Size([16, 10, 6, 94, 94])
        output = output[:, :, :, 7:self.height + 7, 2:self.width + 2]
        if self.verbose:
            print(output.shape) # torch.Size([16, 10, 6, 94, 94])
            print(f'======================================\n\n\n\n')
        return output

    def forward(self, x, mode):
        if mode in ['train','test']:
            y = self.encode(x)
            # print(y.shape) #torch.Size([16, 256, 24, 24])
            return self.decode(y)
        elif mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)

