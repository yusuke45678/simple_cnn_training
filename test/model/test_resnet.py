import random
import torch
from model import model_factory, ModelInfo, ModelOutput, BaseModel


def test_resnet():
    """test resnet models"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in ['resnet18', 'resnet50']:
        for use_pretrained in [True, False]:
            model_info = ModelInfo(
                model_name=model_name,
                n_classes=random.choice([2, 10, 100, 1000]),
                use_pretrained=use_pretrained,
                device=device,
            )
            model = model_factory(model_info)
            assert isinstance(model, BaseModel)

            batch_size = random.randint(1, 16)
            data = torch.rand(batch_size, 3, 224, 224, device=device)  # BCHW
            labels = torch.randint(0, model_info.n_classes - 1, (batch_size, ), device=device)
            output = model(data, labels)
            assert isinstance(output, ModelOutput)
            assert output.logits.shape == (batch_size, model_info.n_classes)
            assert output.loss.shape == (1,)
