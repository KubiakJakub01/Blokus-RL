from .dumbnet import DumbNet
from .blokus_nnet import DCNNet, ResNet


def get_model(model_type):
    model_dict = {
        "dumbnet": DumbNet,
        "dcnnet": DCNNet,
        "resnet": ResNet,
    }
    return model_dict[model_type]
