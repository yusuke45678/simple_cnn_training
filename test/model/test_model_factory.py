import pytest
import torch

from model import configure_model, ModelConfig, BaseModel


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = configure_model(ModelConfig(
        model_name=model_name,
        n_classes=n_classes,
        use_pretrained=use_pretrained,
    ))
    model.to(device)
    assert isinstance(model, BaseModel)
