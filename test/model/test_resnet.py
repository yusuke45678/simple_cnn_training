import random
import torch
import pytest

from model import model_factory, ModelInfo, ModelOutput, BaseModel


@pytest.mark.parametrize('model_name', ['resnet18', 'resnet50'])
@pytest.mark.parametrize('use_pretrained', [True, False])
def test_resnet(
    model_name,
    use_pretrained
):
    """test outputs of resnet models"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = random.choice([2, 10, 100, 1000])

    model = model_factory(ModelInfo(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
        device=device,
    ))
    assert isinstance(model, BaseModel)

    batch_size = random.randint(1, 16)
    data = torch.rand(batch_size, 3, 224, 224, device=device)  # BCHW
    labels = torch.randint(0, n_classes - 1, (batch_size, ), device=device)
    output = model(data, labels)
    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (batch_size, n_classes)
    assert output.loss.ndim == 0
