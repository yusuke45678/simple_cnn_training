import pytest

import torch
from torch import nn

from model import (
    configure_model,
    ModelConfig,
    ModelOutput,
    ClassificationBaseModel,
    get_device,
    get_innermodel,
)


@pytest.mark.parametrize('model_name', ['resnet18', 'resnet50'])
@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('crop_size', [224])
def test_resnet_output(
    model_name,
    use_pretrained,
    n_classes,
    batch_size,
    crop_size,
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

    data = torch.rand(batch_size, 3, crop_size, crop_size, device=device)  # BCHW
    labels = torch.randint(0, n_classes - 1, (batch_size, ), device=device)

    output = model(data, labels)
    assert isinstance(output, ModelOutput)

    assert isinstance(output.logits, torch.Tensor)
    assert output.logits.shape == (batch_size, n_classes)

    assert isinstance(output.loss, torch.Tensor)
    assert output.loss.ndim == 0

    # no labels
    output = model(data)
    assert isinstance(output, ModelOutput)

    assert isinstance(output.logits, torch.Tensor)
    assert output.logits.shape == (batch_size, n_classes)

    assert output.loss is None


@pytest.mark.parametrize('model_name', ['resnet18', 'resnet50'])
@pytest.mark.parametrize('use_pretrained', [True])
@pytest.mark.parametrize('n_classes', [10])
def test_resnet_methods(
    model_name,
    use_pretrained,
    n_classes,
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

    assert isinstance(get_innermodel(model), nn.Module)
    assert isinstance(next(model.parameters()), nn.Parameter)

    model.train()
    assert model.model.training
    model.eval()
    assert not model.model.training
    model.train()
    assert model.model.training

    model.to(torch.device('cpu'))
    assert get_device(model) == torch.device('cpu')

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    model.to(device)
    assert get_device(model) == device


@pytest.mark.parametrize('model_name', ['resnet18', 'resnet50'])
@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('crop_size', [224])
def test_resnet_loss_backward(
    model_name,
    use_pretrained,
    n_classes,
    batch_size,
    crop_size,
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

    data = torch.rand(batch_size, 3, crop_size, crop_size, device=device)  # BCHW
    labels = torch.randint(0, n_classes - 1, (batch_size, ), device=device)

    output = model(data, labels)
    output.loss.backward()
