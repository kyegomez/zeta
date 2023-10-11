import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange


def yolo(input, num_classes, num_anchors, anchors, stride_h, stride_w):
    raw_predictions = rearrange(
        input,
        "b (anchor prediction) h w -> prediction b anchor h w",
        anchor=num_anchors,
        prediction=5 + num_classes,
    )
    anchors = torch.FloatTensor(anchors).to(input.device)
    anchor_sizes = rearrange(anchors, "anchor dim -> dim () anchor () ()")

    _, _, _, in_h, in_w = raw_predictions.shape
    grid_h = rearrange(torch.arange(in_h).float(), "h -> () () h ()").to(input.device)
    grid_w = rearrange(torch.arange(in_w).float(), "w -> () () () w").to(input.device)

    predicted_bboxes = torch.zeros_like(raw_predictions)
    predicted_bboxes[0] = (raw_predictions[0].sigmoid() + grid_w) * stride_w  # center x
    predicted_bboxes[1] = (raw_predictions[1].sigmoid() + grid_h) * stride_h  # center y
    predicted_bboxes[2:4] = (
        raw_predictions[2:4].exp()
    ) * anchor_sizes  # bbox width and height
    predicted_bboxes[4] = raw_predictions[4].sigmoid()  # confidence
    predicted_bboxes[5:] = raw_predictions[5:].sigmoid()  # class predictions
    # merging all predicted bboxes for each image
    return rearrange(
        predicted_bboxes, "prediction b anchor h w -> b (anchor h w) prediction"
    )
