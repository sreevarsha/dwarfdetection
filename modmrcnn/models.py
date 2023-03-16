import os
import torch
import torch.nn as nn
import torchvision
import yaml

from collections import OrderedDict

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator
from cirrus.scale import get_scale, ScaleParallel
from quicktorch.modules.attention.models import AttentionMS
from quicktorch.modules.attention.loss import GuidedAuxLoss
from modmrcnn.forward import my_forward
from modmrcnn.contaminants import ContaminantClassifier, ContaminantSegmenter, ContaminantAloneSegmenter


MaskRCNN.forward = my_forward


def create_maskrcnn_with_backbone(backbone, num_classes):
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    feat_maps = 3
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),) * feat_maps,
        aspect_ratios=((0.5, 1.0, 2.0),) * feat_maps
    )

    model = MaskRCNN(
        backbone, num_classes,
        1024, 1024, [.340, .398], [.757, .923],
        anchor_generator,
        box_roi_pool=roi_pooler,
        box_detections_per_img=50
    )
    model.aux_loss = GuidedAuxLoss()

    size = 1024
    model.transform = GeneralizedRCNNTransform(
        size, size, [.340, .398], [.757, .923], fixed_size=(size, size)
    )

    model.forward = my_forward

    return model


def get_pretrained_model2311(num_classes):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,weights='DEFAULT')

    model = change_first_layer(model)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    size = 512
    model.transform = GeneralizedRCNNTransform(
        size, size, [.340, .398], [.757, .923], fixed_size=(size, size)
    )

    return model
    
    
def playwith_model2311(num_classes):    

    anchor_sizes = ((2,), (4,), (8,), (16,), (32,))

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    )
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,anchor_generator=anchor_generator)

    model = change_first_layer(model)
    

    
    model.rpn.anchor_generator = anchor_generator
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    size = 512
    model.transform = GeneralizedRCNNTransform(
        size, size, [.340, .398], [.757, .923], fixed_size=(size, size)
    )

    return model


def get_pretrained_with_contaminant(num_classes):
    model = get_pretrained_model2311(num_classes - 1)

    model.contaminant_classifier = ContaminantClassifier(256, 5, output_size=7)
    model.contaminant_segmenter = ContaminantSegmenter(256, 5, 8)

    return model


def get_pretrained_with_alone_contaminant(num_classes):
    model = get_pretrained_model2311(num_classes - 1)

    model.contaminant_segmenter = ContaminantAloneSegmenter(256, 5, 8)

    return model


def get_pretrained_with_alone_contaminant2(num_classes):
    model = get_pretrained_model2311(num_classes - 1)

    model.contaminant_segmenter = ContaminantAloneSegmenter(256, 5, 32)

    attention_net_path = "./models/contaminant_segmenter/attention256_5_32.pt"
    attention_weights = torch.load(attention_net_path)
    model.contaminant_segmenter.attention_net.load_state_dict(attention_weights)

    mask_generator_path = "./models/contaminant_segmenter/mask_generator5_32_1.pt"
    mask_generator_weights = torch.load(mask_generator_path)
    model.contaminant_segmenter.mask_generator.load_state_dict(mask_generator_weights)

    return model


def change_first_layer(model):
    conv1 = model.backbone.body.conv1
    first_layer = nn.Sequential(OrderedDict([
        ('scale', ScaleParallel(2)),
        ('conv', nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False))
    ]))
    with torch.no_grad():
        first_layer.conv.weight[:, :3] = conv1.weight
    model.backbone.body.conv1 = first_layer
    return model


def basic_att_model(num_classes):
    backbone = AttentionMS(4, 64, scale=ScaleParallel(2), rcnn=True)
    backbone.out_channels = 128
    return create_maskrcnn_with_backbone(backbone, num_classes)


MODELS = {
    '2311Attention': basic_att_model,
    '2311Pretrained': get_pretrained_model2311,
    '2301PreContaminant': get_pretrained_with_contaminant,
    '0504PreContaminant': get_pretrained_with_alone_contaminant,
    '1605PreContaminant': get_pretrained_with_alone_contaminant2,
    'play_2311'			: playwith_model2311
}


def load_model_from_config(model_key, num_classes):
    config_path = os.path.join('./configs', f'{model_key}.yaml')
    config = load_config(config_path)
    config['scale'] = get_scale(config['scale'])


def get_model(model_key, num_classes):
    if model_key in MODELS:
        return MODELS[model_key](num_classes)
    return load_model_from_config(model_key, num_classes)


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config
