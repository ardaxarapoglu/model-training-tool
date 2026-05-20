"""Factory for building CNN regression models (single float output)."""
import torch
import torch.nn as nn
import torchvision.models as tvm


ARCHITECTURES = {
    "ResNet-18": "resnet18",
    "ResNet-34": "resnet34",
    "ResNet-50": "resnet50",
    "ResNet-101": "resnet101",
    "VGG-16": "vgg16",
    "VGG-19": "vgg19",
    "DenseNet-121": "densenet121",
    "DenseNet-169": "densenet169",
    "MobileNet-V2": "mobilenet_v2",
    "MobileNet-V3-Small": "mobilenet_v3_small",
    "MobileNet-V3-Large": "mobilenet_v3_large",
    "EfficientNet-B0": "efficientnet_b0",
    "EfficientNet-B3": "efficientnet_b3",
    "EfficientNet-B7": "efficientnet_b7",
}


def build_model(model_cfg: dict) -> nn.Module:
    mode = model_cfg.get("mode", "transfer")
    if mode == "transfer":
        return _build_transfer(model_cfg.get("transfer", {}))
    return _build_scratch(model_cfg.get("scratch", {}))


def _build_transfer(cfg: dict) -> nn.Module:
    arch_key = cfg.get("architecture", "ResNet-50")
    arch_id = ARCHITECTURES.get(arch_key, arch_key)
    pretrained = cfg.get("pretrained", True)
    dropout = float(cfg.get("dropout", 0.5))

    weights = "DEFAULT" if pretrained else None

    try:
        model = getattr(tvm, arch_id)(weights=weights)
    except TypeError:
        model = getattr(tvm, arch_id)(pretrained=pretrained)

    # Replace final layer with regression head
    if arch_id.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    elif arch_id.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    elif arch_id.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    elif arch_id.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    elif arch_id.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    else:
        raise ValueError(f"Unsupported architecture: {arch_id}")

    # Freeze backbone if requested
    freeze_backbone = cfg.get("freeze_backbone", False)
    unfreeze_last_n = int(cfg.get("unfreeze_last_n", 0))

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze head (always)
        _unfreeze_head(model, arch_id)

    if unfreeze_last_n > 0:
        _unfreeze_last_n_layers(model, unfreeze_last_n)

    return model


def _unfreeze_head(model, arch_id):
    if arch_id.startswith("resnet"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif arch_id.startswith("vgg"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif arch_id.startswith("densenet"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif arch_id.startswith(("mobilenet", "efficientnet")):
        for p in model.classifier.parameters():
            p.requires_grad = True


def _unfreeze_last_n_layers(model, n):
    params = list(model.parameters())
    for p in params[-n:]:
        p.requires_grad = True


def _build_scratch(cfg: dict) -> nn.Module:
    num_blocks = int(cfg.get("num_conv_blocks", 4))
    base_filters = int(cfg.get("base_filters", 32))
    fc_layers = cfg.get("fc_layers", [256, 128])
    use_bn = cfg.get("batch_norm", True)
    dropout = float(cfg.get("dropout", 0.5))

    layers = []
    in_ch = 3
    for i in range(num_blocks):
        out_ch = base_filters * (2 ** i)
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        in_ch = out_ch

    conv_body = nn.Sequential(*layers)
    fc_head_layers = [nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten()]
    fc_in = in_ch * 4 * 4
    for neurons in fc_layers:
        fc_head_layers.append(nn.Linear(fc_in, int(neurons)))
        fc_head_layers.append(nn.ReLU(inplace=True))
        fc_head_layers.append(nn.Dropout(dropout))
        fc_in = int(neurons)
    fc_head_layers.append(nn.Linear(fc_in, 1))

    return nn.Sequential(conv_body, *fc_head_layers)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
