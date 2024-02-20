import pytest

import torch
from torch import nn

from model import configure_model, ModelConfig, ModelOutput, BaseModel


@pytest.mark.parametrize('model_name', ['abn_r50'])
@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('crop_size', [224])
def test_abn_output(
    model_name,
    use_pretrained,
    n_classes,
    batch_size,
    crop_size,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
        device=device,
    ))
    assert isinstance(model, BaseModel)

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


@pytest.mark.parametrize('model_name', ['abn_r50'])
@pytest.mark.parametrize('use_pretrained', [True])
@pytest.mark.parametrize('n_classes', [10])
def test_abn_methods(
    model_name,
    use_pretrained,
    n_classes,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
        device=device,
    ))

    assert isinstance(model.get_model(), nn.Module)
    assert isinstance(next(model.parameters()), nn.Parameter)

    model.train()
    assert model.model.training
    model.eval()
    assert not model.model.training
    model.train()
    assert model.model.training

    model.to(torch.device('cpu'))
    assert model.get_device() == torch.device('cpu')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda:0'):
        model.to(device)
        assert model.get_device() == device


@pytest.mark.parametrize('model_name', ['abn_r50'])
@pytest.mark.parametrize('use_pretrained', [True, False])
@pytest.mark.parametrize('n_classes', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('crop_size', [224])
def test_zero_output_loss_backward(
    model_name,
    use_pretrained,
    n_classes,
    batch_size,
    crop_size,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
        device=device,
    ))
    assert isinstance(model, BaseModel)

    data = torch.rand(batch_size, 3, crop_size, crop_size, device=device)  # BCHW
    labels = torch.randint(0, n_classes - 1, (batch_size, ), device=device)

    output = model(data, labels)
    output.loss.backward()
