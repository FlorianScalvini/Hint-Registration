import torch
import torch.nn as nn

class SSIM3d(nn.Module):
    """
    This class implements the Structural Similarity Index Measure (SSIM) for 3d images.
    The codes are mainly refactored from

    https://github.com/jinh0park/pytorch-ssim-3D/blob/master/pytorch_ssim/
    """

    def __init__(self, window_size: int = 11, size_average: bool = True) -> None:
        super().__init__()

        self.window_size = window_size
        self.size_average = size_average
        self.channels = 1
        self.kernel = self.__create_3d_kernel(window_size, self.channels)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        n_channels = img1.size(1)

        if n_channels == self.channels and self.kernel.data.type == img1.data.type:
            window = self.kernel
        else:
            window = self.__create_3d_kernel(self.window_size, n_channels)

        window = window.to(img1.device)
        window = window.type_as(img1)

        ssim = self.__ssim(img1, img2, window, self.window_size, n_channels, self.size_average)

        return ssim

    def __create_3d_kernel(self, window_size: int, channels: int) -> torch.Tensor:
        kernel1d = self.__gaussian(window_size, 1.5).unsqueeze(1)
        kernel2d = torch.mm(kernel1d, kernel1d.t()).unsqueeze(0).unsqueeze(0)
        kernel3d = torch.mm(kernel1d, kernel2d.reshape(1, -1))
        kernel3d = kernel3d.reshape(window_size, window_size, window_size).unsqueeze(0).unsqueeze(0)

        kernel = kernel3d.expand(channels, 1, window_size, window_size, window_size).contiguous()

        return kernel

    def __gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        kernel = torch.Tensor(
            [torch.exp(-(x - torch.tensor(window_size // 2)) ** 2 / (2 * torch.tensor(sigma ** 2))) for x in
             range(window_size)])
        kernel = kernel / kernel.sum()

        return kernel

    def __ssim(self, img1: torch.Tensor, img2: torch.Tensor, kernel: torch.Tensor,
               kernel_size: int, channels: int, size_average: bool) -> torch.Tensor:
        mu1 = F.conv3d(img1, kernel, padding=kernel_size // 2, groups=channels)
        mu2 = F.conv3d(img2, kernel, padding=kernel_size // 2, groups=channels)

        mu1_squared = mu1 ** 2
        mu2_squared = mu2 ** 2

        mu1mu2 = mu1 * mu2

        sigma1 = F.conv3d(img1 * img1, kernel, padding=kernel_size // 2, groups=channels) - mu1_squared
        sigma2 = F.conv3d(img2 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu2_squared
        sigma12 = F.conv3d(img1 * img2, kernel, padding=kernel_size // 2, groups=channels) - mu1mu2

        const1 = 0.01 ** 2
        const2 = 0.03 ** 2

        numerator = (2 * mu1mu2 + const1) * (2 * sigma12 + const2)
        denominator = (mu1_squared + mu2_squared + const1) * (sigma1 + sigma2 + const2)
        ssim_map = numerator / denominator

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)