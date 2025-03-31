import numpy as np
from torch import nn
import torch
from Forecasting.config import cfg


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, decouple_loss):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        mse = torch.sum(differ ** 2, (2, 3, 4))  # b s
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        mae = torch.mean(mae)  # 1
        loss = mse + mae
        if 'PredRNN-V2' in cfg.model_name:
            decouple_loss = torch.sum(decouple_loss, (1, 3))  # s l b c -> s b
            decouple_loss = torch.mean(decouple_loss)  # 1
            loss = loss + cfg.decouple_loss_weight * decouple_loss
        return loss


class Loss2(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        mse = torch.sum(differ ** 2, (2, 3, 4))  # b s
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        mae = torch.mean(mae)  # 1
        loss = mse + mae
        return loss


class Loss_MAE(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        # print(differ)
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # b s
        mae = torch.mean(mae)  # 1
        return mae


class Loss_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, mask):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        differ[:, :, :, np.where(mask == 0)[0], np.where(mask == 0)[1]] = 0
        #TODO delete if forecasting not only flux
        # differ[:, :, 1:] = 0

        mse = torch.sum(differ ** 2, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse


class Loss_MSE_eigenvalues(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, mask, eigens, alpha = 0.0):
        # print(truth.shape)
        # print(pred.shape)
        # print(eigens.shape)

        differ = truth - pred  # b s c h w
        differ[:, :, :, np.where(mask == 0)[0], np.where(mask == 0)[1]] = 0
        # mse = torch.sum(differ ** 2, (2, 3, 4))
        tmp = (differ ** 2) * (1-alpha) + eigens * alpha
        # print(torch.sum(tmp-differ**2))
        mse = torch.sum(tmp, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse


class Loss_MSE_informed(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, features, mask, alpha = 0.0, beta=0.0, eigenvalues=None):
        # print(truth.shape)
        # print(pred.shape)
        # print(features.shape)

        # print(eigenvalues.shape)

        A_model = features[:, :, :3]
        if cfg.features_amount == 9:
            B_model = features[:, :, 3:6]
        else:
            B_model = torch.zeros((A_model.shape[0], A_model.shape[1], 6, A_model.shape[3], A_model.shape[4]))
            for b in range(B_model.shape[0]):
                for i in range(6):
                    for t in range(features.shape[1]):
                        B_model[b, t, i] = features[b, t, 3 + i] * eigenvalues[b, t, i]

            min_vals, max_vals = cfg.min_vals, cfg.max_vals
            #Flux-Flux, SST-SST, Press-Press, Flux-SST, Flux-Press, SST-Press
            for i in range(3):
                B_model[:, :, i] = (B_model[:, :, i])/((max_vals[i]-min_vals[i])**3)

            # B_model[:, :, 1] /= 100

            B_model[:, :, 3] = (B_model[:, :, 3]) / ((max_vals[0] - min_vals[0])**2 * (max_vals[1] - min_vals[1])**2)
            B_model[:, :, 4] = (B_model[:, :, 4]) / ((max_vals[0] - min_vals[0])**2 * (max_vals[2] - min_vals[2])**2)
            B_model[:, :, 5] = (B_model[:, :, 5]) / ((max_vals[1] - min_vals[1])**2 * (max_vals[2] - min_vals[2])**2)


            for i in range(3):
                B_model[:, :, i] += B_model[:, :, 3+i]

            B_model = B_model[:, :, :3]
            B_model = B_model.cuda()

        # for i in range(3):
        #     print(i)
        #     print(torch.max(B_model[:, :, i]))
        #     print(torch.min(B_model[:, :, i]))
        # raise ValueError

        # TODO flux only?
        # A_model[:, :, 1:] = 0
        # B_model[:, :, 1:] = 0
        # print(A_model.shape)
        # print(B_model.shape)

        differ = truth - pred  # b s c h w
        loss_drift = pred - A_model

        loss_variance = pred - A_model - B_model[:, :, :3]
        # loss_variance = 0
        # raise ValueError
        differ[:, :, :, np.where(mask == 0)[0], np.where(mask == 0)[1]] = 0
        loss_drift[:, :, :, np.where(mask == 0)[0], np.where(mask == 0)[1]] = 0
        loss_variance[:, :, :, np.where(mask == 0)[0], np.where(mask == 0)[1]] = 0

        tmp = (differ ** 2) * (1-alpha) + (loss_variance ** 2) * (alpha)
        # tmp = torch.abs(differ) * alpha + torch.abs(loss_variance) * (1-alpha)

        #TODO delete if forecasting not only flux
        # tmp[:, :, 1:] = 0

        # print(torch.sum(tmp-differ**2))
        mse = torch.sum(tmp, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse