import torch
import torch.nn as nn
from einops import rearrange, repeat

class TCJA(nn.Module):
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=T, out_channels=T,
                              kernel_size=kernel_size_t, padding='same', bias=False)
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        x_c = x.permute(0, 2, 1)
        conv_t_out = self.conv(x).permute(1, 0, 2)
        conv_c_out = self.conv_c(x_c).permute(2, 0, 1)
        out = self.sigmoid(conv_c_out * conv_t_out)
        y_seq = x_seq * out[:, :, :, None, None]
        return y_seq

class STCJA(nn.Module):
    def __init__(self, kernel_size_s: int = 3, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()
        
        self.T = T
        self.conv_t = nn.Conv1d(in_channels=T, out_channels=T,
                                kernel_size=kernel_size_t, padding='same', bias=False)
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding='same', bias=False)
        
        assert kernel_size_s in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size_s == 7 else 1
        self.conv_s = nn.Conv2d(2, 1, kernel_size_s, padding=padding, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        # 时间注意力
        x_t = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        conv_t_out = self.conv_t(x_t).permute(1, 0, 2)

        # 通道注意力
        x_c = x_t.permute(0, 2, 1)
        conv_c_out = self.conv_c(x_c).permute(2, 0, 1)

        # 空间注意力
        x_s = rearrange(x_seq, "t b c h w -> b (t c) h w")  # 将时间和通道维度合并
        avgout = torch.mean(x_s, dim=1, keepdim=True)
        maxout, _ = torch.max(x_s, dim=1, keepdim=True)
        x_s = torch.cat([avgout, maxout], dim=1)  # 维度 (N, 2, H, W)
        conv_s_out = self.conv_s(x_s)
        conv_s_out = rearrange(conv_s_out, "b 1 h w -> 1 b 1 h w").repeat(self.T, 1, 1, 1, 1)  # 恢复时间维度并重复

        conv_c_out = conv_c_out.unsqueeze(3).unsqueeze(4)
        conv_t_out = conv_t_out.unsqueeze(3).unsqueeze(4)

        # 结合注意力
        out = self.sigmoid(conv_c_out * conv_t_out * conv_s_out)

        # 应用注意力掩码
        y_seq = x_seq * out[:, :, :, :, :]  # 调整维度以匹配输入 (N, T, C, H, W)

        return y_seq


class TLA(nn.Module):
    # TODO: 删除无用参数
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()

        # Excitation
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        x_c = x.permute(0, 2, 1)
        conv_c_out = self.conv_c(x_c).permute(2, 0, 1)
        out = self.sigmoid(conv_c_out)
        y_seq = x_seq * out[:, :, :, None, None]
        return y_seq


class CLA(nn.Module):
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=T, out_channels=T,
                              kernel_size=kernel_size_t, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        x = torch.mean(x_seq.permute(1, 0, 2, 3, 4), dim=[3, 4])
        conv_t_out = self.conv(x).permute(1, 0, 2)
        out = self.sigmoid(conv_t_out)
        # max_out = self.con(torch.amax(x_seq, dim =[3,4]))

        y_seq = x_seq * out[:, :, :, None, None]
        return y_seq


class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)