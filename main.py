import torch
import torch.nn as nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def rms_norm_main():
    rows = 100
    cols = 2048
    epsilon = 1e-6

    # 初始化src张量
    src = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) % 4 + 1  # 生成周期性序列
    rms_norm_layer = RMSNorm(cols)
    output = rms_norm_layer(src)
    print(output)


def layer_norm_main():
    rows = 100
    cols = 2048
    epsilon = 1e-5

    # 初始化src张量
    src = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) % 4 + 1  # 生成周期性序列
    # 初始化gamma和beta张量
    gamma = torch.arange(cols, dtype=torch.float32) % 4 + 1 * 0.5
    beta = torch.arange(cols, dtype=torch.float32) % 4 + 1 * 0.5

    # 使用PyTorch的LayerNorm函数
    dst = torch.nn.functional.layer_norm(src, (cols,), weight=gamma, bias=beta, eps=epsilon)

    # 输出处理结果的第一个元素
    print("LayerNorm Output:", dst[0, 0].item())
    print(dst.shape)
    print(dst)


def main():
    rms_norm_main()
    layer_norm_main()


if __name__ == "__main__":
    main()
