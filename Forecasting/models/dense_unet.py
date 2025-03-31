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


class DenseU_Net(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size=2, stride=2, padding='same'):
        super(DenseU_Net, self).__init__()
        self.batch = b_h_w[0]
        self.height = b_h_w[1]
        self.width = b_h_w[2]
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=input_channel, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # dense down
        self.Dense1_2 = nn.Linear(in_features=input_channel, out_features=64)
        self.Dense12_3 = nn.Linear(in_features=input_channel, out_features=128)
        self.Dense23_4 = nn.Linear(in_features=64, out_features=256)
        self.Dense34_5 = nn.Linear(in_features=128, out_features=512)

        # dense up
        self.Dense6_7 = nn.Linear(in_features=512, out_features=256)
        self.Dense67_8 = nn.Linear(in_features=512, out_features=128)
        self.Dense78_9 = nn.Linear(in_features=256, out_features=64)
        self.Dense89_10= nn.Linear(in_features=128, out_features=output_channel)

        # dense middle
        self.dense345_7 = nn.Linear(in_features=fff, out_features=512)
        self.dense234_8 = nn.Linear(in_features=fff, out_features=256)
        self.dense123_9 = nn.Linear(in_features=fff, out_features=128)
        self.dense123_10 = nn.Linear(in_features=fff, out_features=64)

        self.Up7 = up_conv(ch_in=1024, ch_out=512)
        self.Conv7 = conv_block(ch_in=1024, ch_out=512)
        self.Up8 = up_conv(ch_in=512, ch_out=256)
        self.Conv8 = conv_block(ch_in=512, ch_out=256)
        self.Up9 = up_conv(ch_in=256, ch_out=128)
        self.Conv9 = conv_block(ch_in=256, ch_out=128)
        self.Up10 = up_conv(ch_in=128, ch_out=64)
        self.Conv10 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1_x2 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x3 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x4 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x5 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x7_1 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x7_2 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x8_1 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x8_2 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x9_1 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x9_2 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x10_1 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.Conv_1x1_x10_2 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
    # def forward(self, x, m, layer_hiddens, encoder, decoder):
    def forward(self, x):
        # encoding path
        # x = encoder(x)
        x = torch.nn.functional.pad(x, (2, 3, 7, 8))
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x1 = x

        x1_conv = self.Conv1(x)
        x2 = self.Maxpool(x1_conv)
        x2_conv = self.Conv2(x2)
        print(x2.shape)
        x12_dense = self.Dense1_2(x)
        print(x12_dense.shape)
        x2 = torch.cat((x12_dense, x2), dim=1)
        x2 = self.Conv_1x1_x2(x2)
        x2 = self.Conv2(x2)
        print(x2.shape)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x12_3 = self.Dense12_3(torch.cat((x1, x2), dim=1))
        x3 = torch.cat((x12_3, x3), dim=1)
        x3 = self.Conv_1x1_x3(x3)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x23_4 = self.Dense23_4(torch.cat((x2, x3), dim=1))
        x4 = torch.cat((x23_4, x4), dim=1)
        x4 = self.Conv_1x1_x4(x4)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x34_5 = self.Dense34_5(torch.cat((x3, x4), dim=1))
        x5 = torch.cat((x34_5, x5), dim=1)
        x5 = self.Conv_1x1_x5(x5)

        x6 = x5

        # decoding + concat path
        x7_dense = self.dense345_7(torch.cat((x3, x4, x5), dim=1))
        x7_up = self.Up7(x6)
        x7 = torch.cat((x7_dense, x7_up))
        x7 = self.Conv_1x1_x7_1(x7)
        x7 = self.Conv7(x7)
        x6_7 = self.Dense6_7(x6)
        x7 = torch.cat((x7, x6_7), dim=1)
        x7 = self.Conv_1x1_x7_2(x7)

        x8_dense = self.dense234_8(torch.cat((x2, x3, x4), dim=1))
        x8_up = self.Up8(x7)
        x8 = torch.cat((x8_dense, x8_up), dim=1)
        x8 = self.Conv_1x1_x8_1(x8)
        x8 = self.Conv8(x8)
        x67_8 = self.Dense67_8(torch.cat((x6, x7), dim=1))
        x8 = torch.cat((x8, x67_8))
        x8 = self.Conv_1x1_x8_2(x8)

        x9_dense = self.dense123_9(torch.cat((x1, x2, x3), dim=1))
        x9_up = self.Up9(x8)
        x9 = torch.cat((x9_dense, x9_up), dim=1)
        x9 = self.Conv_1x1_x9_1(x9)
        x9 = self.Conv9(x9)
        x78_9 = self.Dense78_9(torch.cat((x7, x8), dim=1))
        x9 = torch.cat((x9, x78_9))
        x9 = self.Conv_1x1_x9_2(x9)

        x10_dense = self.dense123_10(torch.cat((x1, x2, x3), dim=1))
        x10_up = self.Up10(x9)
        x10 = torch.cat((x10_dense, x10_up), dim=1)
        x10 = self.Conv_1x1_x10_1(x10)
        x10 = self.Conv10(x10)
        x89_10 = self.Dense89_10(torch.cat((x8, x9), dim=1))
        x10 = torch.cat((x10, x89_10))
        x10 = self.Conv_1x1_x10_2(x10)

        output = torch.reshape(x10, (-1, cfg.out_len, cfg.features_amount, x10.shape[2], x10.shape[3]))
        output = output[:, :, :, 7:self.height + 7, 2:self.width + 2]
        return output
        # next_layer_hiddens = []
        # x = decoder(d1)
        # decouple_loss = torch.zeros([cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state]).cuda()
        # return x, m, next_layer_hiddens, decouple_loss
