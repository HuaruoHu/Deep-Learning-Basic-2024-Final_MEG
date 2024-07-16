import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np



    
class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return X
    
    

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        input_window_samples=None,
        n_filters_time=20,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=75,
        pool_time_stride=15,
        final_conv_length=13,
        pool_mode="mean",
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        drop_prob=0.5,
    ) -> None:
        super().__init__()
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_classes = 1854

        self.conv1 = nn.Conv2d(1, self.n_filters_time, (self.filter_time_length, 1), stride=1)
        self.conv2 = nn.Conv2d(self.n_filters_time,self.n_filters_spat,(1,self.in_dim), stride=1,bias=not self.batch_norm)
        self.norm3 = nn.BatchNorm2d(self.n_filters_spat, momentum=self.batch_norm_alpha, affine=True, track_running_stats=True)
        self.avgp4 = nn.AvgPool2d(kernel_size=(self.pool_time_length, 1), stride=(self.pool_time_stride, 1), padding=0)
        self.drop5 = nn.Dropout(p=self.drop_prob, inplace=False)
        self.conv6 = nn.Conv2d(self.n_filters_spat, self.n_classes, (self.final_conv_length, 1), stride=(1, 1), bias=True)
        self.soft7 = nn.LogSoftmax(dim=1)

    def ensure4d(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x
    
    def square(self,x):
        return x * x

    def safe_log(self,x, eps=1e-6):
        return torch.log(torch.clamp(x, min=eps))

    def transpose_time_to_spat(self,x):
        return x.permute(0, 3, 2, 1)

    def np_to_th(self,
        X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs
    ):
        if not hasattr(X, "__len__"):
            X = [X]
        X = np.asarray(X)
        if dtype is not None:
            X = X.astype(dtype)
        X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
        if pin_memory:
            X_tensor = X_tensor.pin_memory()
        return X_tensor

    def squeeze_final_output(self,x):
        assert x.size()[3] == 1
        x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ensure4d(x)
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.square(x)
        x = self.avgp4(x)
        x = self.safe_log(x, eps=1e-6)
        x = self.drop5(x)
        x = self.conv6(x)
        x = self.soft7(x)
        x = self.squeeze_final_output(x)
        return x