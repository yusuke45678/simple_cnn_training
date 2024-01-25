import os

from model import X3D, ResNet50, ResNet18, BaseModel, ModelConfig


def configure_model(
        model_info: ModelConfig
) -> BaseModel:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        BaseModel: model
    """

    if model_info.use_pretrained:
        # Specity the directory where a pre-trained model is stored.
        # Otherwise, by default, models are stored in users home dir `~/.torch`
        os.environ['TORCH_HOME'] = model_info.torch_home

    if model_info.model_name == 'resnet18':
        model = ResNet18(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'resnet50':
        model = ResNet50(model_info)  # type: ignore[assignment]

    elif model_info.model_name == 'x3d':
        model = X3D(model_info)  # type: ignore[assignment]

    else:
        raise ValueError('invalid model_info.model_name')

    model = model.to(model_info.device)

    if model_info.gpu_strategy == 'dp':
        model.set_data_parallel()

    return model
