import copy
from typing import Optional

import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d

from model import ModelConfig, ClassificationBaseModel, ModelOutput


class ABNResNet50(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.prepare_model()

    def prepare_model(self):
        self.model = ABNResNet50Model(
            self.model_config.n_classes,
            self.model_config.use_pretrained
        )

    def __call__(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        logits_x, logits_ax, _ = self.model(pixel_values)

        if labels is None:
            return ModelOutput(logits=logits_x)

        loss_x = self.criterion(logits_x, labels)
        loss_ax = self.criterion(logits_ax, labels)
        loss = loss_x + loss_ax

        return ModelOutput(
            logits=logits_x,
            loss=loss
        )


class ABNResNet50Model(nn.Module):
    def __init__(self, n_classes, use_pretrained):
        super().__init__()

        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar100_resnet56",
            pretrained=use_pretrained,
        )

        self.resnet50_bottom = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
        )
        r50b_num_features = self.get_last_num_features(self.resnet50_bottom)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            n_classes
        )

        self.resnet50_top = nn.Sequential(
            self.model.avgpool,
            nn.Flatten(),
            self.model.fc
        )

        self.attention_branch1 = nn.Sequential(
            # reusing layer3 block structure
            #     (except the first block with downsampling)
            copy.deepcopy(self.model.layer3[1:]),

            nn.BatchNorm2d(r50b_num_features),
            nn.Conv2d(
                r50b_num_features,
                n_classes,
                kernel_size=1
            ),
            nn.ReLU(inplace=True)
        )
        self.attention_branch2 = nn.Sequential(
            nn.Conv2d(
                n_classes,
                1,
                kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.attention_branch3 = nn.Sequential(
            nn.Conv2d(
                n_classes,
                n_classes,
                kernel_size=1),
        )
        self.attention = None

    def get_last_num_features(self, module: nn.Module):
        return list(module.modules())[-1].num_features

    def get_attention(self):
        return self.attention

    def gap(self, x: torch.Tensor):
        x = adaptive_avg_pool2d(x, (1, 1))  # BCHW -> BC11
        x = x.squeeze((-2, -1))  # BC11 -> BC
        return x

    def forward(self, x):
        x1 = self.resnet50_bottom(x)

        ax1 = self.attention_branch1(x1)
        self.attention = self.attention_branch2(ax1)

        x2 = x1 * self.attention
        x3 = self.resnet50_top(x2)

        ax2 = self.attention_branch3(ax1)  # B,C,H,W
        ax3 = self.gap(ax2)  # B,C,1,1 -> B,C

        return x3, ax3, self.attention
