import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# from models.xception import return_pytorch04_xception

from model import Xception


def DCT_mat(size):
    m = [
        [
            (np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
            * np.cos((j + 0.5) * np.pi * i / size)
            for j in range(size)
        ]
        for i in range(size)
    ]
    return m


def generate_filter(start, end, size):
    return [
        [0.0 if i + j > end or i + j <= start else 1.0 for j in range(size)]
        for i in range(size)
    ]


def norm_sigma(x):
    return 2.0 * torch.sigmoid(x) - 1.0


class Filter(nn.Module):
    def __init__(
            self,
            img_size: int,
            band_start: float,
            band_end: float,
            use_learnable: bool = True,
            norm: bool = False,
    ) -> None:
        super(Filter, self).__init__()

        self.use_learnable = use_learnable

        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, img_size)),
            requires_grad=False,
        )

        if use_learnable:
            self.learnable = nn.Parameter(
                torch.randn(img_size, img_size), requires_grad=True
            )
            self.learnable.data.normal_(0, 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(
                    torch.tensor(generate_filter(band_start, band_end, img_size))
                ),
                requires_grad=False,
            )

    def forward(self, x):

        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt

        return y


class FAD_Head(nn.Module):
    def __init__(self, img_size: int) -> None:
        super(FAD_Head, self).__init__()

        # initialize DCT matrix
        self._DCT_all = nn.Parameter(
            torch.tensor(DCT_mat(img_size)).float(), requires_grad=False
        )
        self._DCT_all_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(img_size)).float(), 0, 1),
            requires_grad=False
        )

        # Define base filter and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(img_size, 0, img_size // 16)
        middle_filter = Filter(img_size, img_size // 16, img_size // 8)
        high_filter = Filter(img_size, img_size // 8, img_size)
        all_filter = Filter(img_size, 0, img_size * 2)

        self.filters = nn.ModuleList(
            [low_filter, middle_filter, high_filter, all_filter]
        )

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        x_freq = torch.nan_to_num(x_freq, nan=0.0)  # NaN을 0으로 대체
        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)

            x_freq = torch.nan_to_num(x_freq, nan=0.0)  # NaN을 0으로 대체
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y = torch.nan_to_num(y, nan=0.0)  # NaN을 0으로 대체
            y_list.append(y)

        out = torch.cat(y_list, dim=1)
        out = torch.nan_to_num(out, nan=0.0)

        return out


class LFS_Head(nn.Module):
    def __init__(self, img_size: int, window_size: int, M: int) -> None:
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self.M = M

        # initialize DCT matrix
        self._DCT_patch = nn.Parameter(
            torch.tensor(DCT_mat(window_size)).float(), requires_grad=False
        )
        self._DCT_patch_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1),
            requires_grad=False,
        )

        self.unfold = nn.Unfold(
            kernel_size=(window_size, window_size), stride=2, padding=4
        )

        # initialize filters
        self.filters = nn.ModuleList(
            [
                Filter(
                    window_size,
                    window_size * 2.0 / M * i,
                    window_size * 2.0 / M * (i + 1),
                    norm=True,
                )
                for i in range(M)
            ]
        )

    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.0) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8) // 2) + 1
        assert size_after == 149, "Size after unfold must be 149."
        # 주어진 이미지 너비에서 슬라이딩 윈도우의 크기와 추가적인 패딩을 고려하여, 최종적으로 남는 패치의 수를 계산

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)  # [N, C*S*S, L] L = Number of block

        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]

        x_dct = self._DCT_patch @ x_unfold

        x_dct = x_dct @ self._DCT_patch_T  # [N, L, C, S, S]

        # M kernels filtering
        y_list = []
        for i in range(self.M):
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)

            y = self.filters[i](y)

            y = torch.sum(y, dim=(2, 3, 4))

            y = y.reshape(N, size_after, size_after).unsqueeze(1)

            y_list.append(y)

        out = torch.cat(y_list, dim=1)

        return out


class MixBlock(nn.Module):
    def __init__(self, c_in: int, width: int, height: int) -> None:
        super(MixBlock, self).__init__()

        self.FAD_query = nn.Conv2d(c_in, c_in, kernel_size=(1, 1))
        self.LFS_query = nn.Conv2d(c_in, c_in, kernel_size=(1, 1))

        self.FAD_key = nn.Conv2d(c_in, c_in, kernel_size=(1, 1))
        self.LFS_key = nn.Conv2d(c_in, c_in, kernel_size=(1, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, kernel_size=(1, 1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)

        self.LFS_conv = nn.Conv2d(c_in, c_in, kernel_size=(1, 1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H, "Width and Height must be equal."

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)  # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)  # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  # [BC, W, W]

        energy_stable = energy - energy.max(dim=-1, keepdim=True)[0]  # 각 행에서 최대값을 뺌
        attention = self.softmax(energy_stable).view(B, C, W, W)
        ####
        # attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))

        return y_FAD, y_LFS


class F3Net(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,
            img_width: int = 299,
            img_height: int = 299,
            LFS_window_size: int = 10,  # lfs 모듈 크기
            LFS_M: int = 6,
    ) -> None:  # 필터 수
        super(F3Net, self).__init__()

        assert img_width == img_height, "Image width and height must be equal."
        self.img_size = img_width
        self._LFS_window_size = LFS_window_size
        self._LFS_M = LFS_M
        self.num_classes = num_classes

        self.fad_head = FAD_Head(self.img_size)

        self.lfs_head = LFS_Head(self.img_size, self._LFS_window_size, self._LFS_M)

        ###############
        self._init_xcep_fad()
        self._init_xcep_lfs()

        self.mix_block7 = MixBlock(c_in=728, width=19, height=19)
        self.mix_block12 = MixBlock(c_in=1024, width=10, height=10)
        self.excep_forwards = [
            "conv1",
            "bn1",
            "relu",
            "conv2",
            "bn2",
            "relu",
            "block1",
            "block2",
            "block3",
            "block4",
            "block5",
            "block6",
            "block7",
            "block8",
            "block9",
            "block10",
            "block11",
            "block12",
            "conv3",
            "bn3",
            "relu",
            "conv4",
            "bn4",
        ]

        # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096, num_classes)
        self.dp = nn.Dropout(p=0.2)

    def forward(self, x):

        fad_input = self.fad_head(x)
        lfs_input = self.lfs_head(x)

        x_fad, x_fls = self._features(fad_input, lfs_input)
        x_fad = self._norm_feature(x_fad)
        x_fls = self._norm_feature(x_fls)

        x_cat = torch.cat((x_fad, x_fls), dim=1)
        x_drop = self.dp(x_cat)
        logit = self.fc(x_drop)
        return logit

    def _features(self, x_fad, x_fls):
        for forward_func in self.excep_forwards:
            x_fad = getattr(self.FAD_xcep, forward_func)(x_fad)
            x_fls = getattr(self.LFS_xcep, forward_func)(x_fls)
            if torch.isnan(x_fls).any():
                print(f"{forward_func}")
                exit(0)

            if forward_func == "block7":
                x_fad, x_fls = self.mix_block7(x_fad, x_fls)
            if forward_func == "block12":
                x_fad, x_fls = self.mix_block12(x_fad, x_fls)

        return x_fad, x_fls

    def _norm_feature(self, x):
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x

    def _init_xcep_fad(self):

        self.FAD_xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        conv1_data = state_dict['conv1.weight'].data

        self.FAD_xcep.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        self.FAD_xcep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            self.FAD_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / 4.0

    def _init_xcep_lfs(self):
        self.LFS_xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        conv1_data = state_dict['conv1.weight'].data

        self.LFS_xcep.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        self.LFS_xcep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
        for i in range(int(self._LFS_M / 3)):
            self.LFS_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / float(self._LFS_M / 3.0)


def _training_config(args):
    # Dataset Argument
    args.num_channels = 3
    args.num_classes = 1
    args.image_size = 299
    args.metric_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

    # Training Argument
    args.train_batch_size = 16  # 12

    args.train_frame_num = 5  # 300
    args.test_batch_size = 128  # 64
    args.test_frame_num = 50
    args.final_epoch = 1  # 5

    # Optimizer Argument
    args.optimizer_name = 'Adam'
    args.lr = 0.0002
    args.weight_decay = 1e-4
    args.adjust_learning_rate = adjust_learning_rate

    return args


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""

    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def adjust_learning_rate(optimizer, epochs, train_loader_len, learning_rate):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            epochs * train_loader_len,
            0.002,  # lr_lambda computes multiplicative factor
            1e-6 / learning_rate))

    return scheduler


def get_xcep_state_dict(pretrained_path='./xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict
