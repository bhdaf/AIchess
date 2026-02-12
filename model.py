"""
简化策略价值网络模型

使用较小的网络结构，适合在普通机器上快速训练和推理。
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple_chess_ai.game import NUM_ACTIONS, BOARD_HEIGHT, BOARD_WIDTH


def get_device():
    """获取计算设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class PolicyValueNet(nn.Module):
    """
    策略价值网络

    简化版本：
    - 输入: 14x10x9
    - 特征提取: 128通道, 4个残差块
    - 策略头: 输出所有走法的概率
    - 价值头: 输出局面评估 [-1, 1]
    """

    def __init__(self, num_channels=128, num_res_blocks=4):
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # 初始卷积
        self.input_conv = nn.Conv2d(14, num_channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, 4, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * BOARD_HEIGHT * BOARD_WIDTH, NUM_ACTIONS)

        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * BOARD_HEIGHT * BOARD_WIDTH, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 初始卷积
        x = F.relu(self.input_bn(self.input_conv(x)))

        # 残差块
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = F.softmax(self.policy_fc(p), dim=1)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class ChessModel:
    """
    模型管理类

    处理模型的构建、保存、加载和预测。
    """

    def __init__(self, num_channels=128, num_res_blocks=4):
        self.device = get_device()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.model = None

    def build(self):
        """构建网络"""
        self.model = PolicyValueNet(self.num_channels, self.num_res_blocks)
        self.model = self.model.to(self.device)
        self.model.eval()

    def save(self, path):
        """保存模型权重和配置"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        config_path = path.replace('.pth', '_config.json')
        config = {
            'num_channels': self.num_channels,
            'num_res_blocks': self.num_res_blocks,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """加载模型权重"""
        if not os.path.exists(path):
            return False
        config_path = path.replace('.pth', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.num_channels = config.get('num_channels', 128)
            self.num_res_blocks = config.get('num_res_blocks', 4)
        if self.model is None:
            self.build()
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return True

    def predict(self, planes):
        """
        预测走法概率和局面评估

        Args:
            planes: numpy array, shape (14, 10, 9) 或 (batch, 14, 10, 9)

        Returns:
            policy: numpy array, shape (NUM_ACTIONS,)
            value: float
        """
        if len(planes.shape) == 3:
            planes = np.expand_dims(planes, 0)
        tensor = torch.FloatTensor(planes).to(self.device)
        with torch.no_grad():
            policy, value = self.model(tensor)
        return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]
