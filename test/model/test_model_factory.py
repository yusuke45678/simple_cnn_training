import pytest

import torch
from torch import nn

from model import (
    configure_model,
    ModelConfig,
    ClassificationBaseModel,
    get_innermodel,
)


@pytest.mark.parametrize(
    'model_name',
    ["resnet18", "resnet50", "abn_r50", "vit_b", "x3d", "zero_output_dummy"]
)
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('use_pretrained', [True, False])
def test_model_factory(
    model_name,
    n_classes,
    use_pretrained,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    model.to(device)
    assert isinstance(model, ClassificationBaseModel)


@pytest.mark.parametrize(
    'model_name',
    ["resnet18", "resnet50", "abn_r50", "vit_b", "x3d", "zero_output_dummy"]
)
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('use_pretrained', [True, False])
def test_model_data_parallel(
    model_name,
    n_classes,
    use_pretrained,
):
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    model.to(device)
    dp_model = nn.DataParallel(model)
    isinstance(dp_model, nn.DataParallel)
    assert isinstance(dp_model.module, ClassificationBaseModel)
    assert isinstance(get_innermodel(dp_model.module), nn.Module)
