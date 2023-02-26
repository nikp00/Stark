import argparse
import torch
import _init_paths
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_stark_lightning_x_trt
from lib.config.stark_lightning_X_trt.config import cfg, update_config_from_file
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
from typing import Tuple
from lib.test.evaluation.environment import env_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Parse args for training")
    parser.add_argument(
        "--script", type=str, default="stark_lightning_X_trt", help="script name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baseline_rephead_4_lite_search5",
        help="yaml configure file name",
    )
    args = parser.parse_args()
    return args


def get_data(bs, sz, dtype=torch.float16):
    img_patch = torch.randn(bs, 3, sz, sz, requires_grad=True, dtype=dtype)
    mask = torch.rand(sz, sz, requires_grad=True, dtype=dtype) > 0.5
    return img_patch, mask


class Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).reshape(
            (1, 3, 1, 1)
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).reshape(
            (1, 3, 1, 1)
        )

    def forward(
        self, patch: torch.Tensor, amask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_4d = (patch / 255.0 - self.mean) / self.std  # (1, 3, H, W)

        # Deal with the attention mask
        amask_3d = amask.unsqueeze(0)  # (1,H,W)
        return patch_4d, amask_3d.to(torch.bool)


class MaskModel(nn.Module):
    def __init__(self, size=128):
        super().__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.size = size

    def forward(self, in_img: torch.Tensor) -> torch.Tensor:
        in_img = in_img.to(torch.float)

        mask = in_img.mean(dim=1) > 0
        mask = mask.to(torch.float)
        # mask = mask.to(torch.float) * 255
        mask = mask.unsqueeze(0)
        mask = self.max_pool2d(mask)
        # mask = torch.clamp(mask, 0, 1)
        mask = mask.squeeze(0).squeeze(0)
        mask = mask.to(torch.bool)
        mask = torch.bitwise_not(mask)

        return mask


class Backbone_Bottleneck_PE(nn.Module):
    def __init__(self, backbone, bottleneck, position_embed):
        super(Backbone_Bottleneck_PE, self).__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.position_embed = position_embed
        self.preprocessor = Preprocessor()
        self.mask_nn = MaskModel()

    def forward(self, img: torch.Tensor):
        mask = self.mask_nn(img)
        img, mask = self.preprocessor(img, mask)

        feat = self.bottleneck(self.backbone(img))  # BxCxHxW
        mask_down = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(
            torch.bool
        )[0]
        pos_embed = self.position_embed(
            1
        )  # 1 is the batch-size. output size is BxCxHxW
        # adjust shape
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask_down.flatten(1)  # BxHW
        return feat_vec, mask_vec, pos_embed_vec


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "backbone_bottleneck_pe.onnx"
    """update cfg"""
    args = parse_args()
    yaml_fname = "experiments/%s/%s.yaml" % (args.script, args.config)
    update_config_from_file(yaml_fname)
    """set some values"""
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    # build the stark model
    model = build_stark_lightning_x_trt(cfg, phase="test")
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = os.path.join(
            save_dir,
            "checkpoints/train/%s/%s/STARKLightningXtrt_ep0500.pth.tar"
            % (args.script, args.config),
        )
        model.load_state_dict(
            torch.load(checkpoint_name, map_location="cpu")["net"], strict=True
        )
    # transfer to test mode
    model = repvgg_model_convert(model)
    model.eval()
    """ rebuild the inference-time model """
    backbone = model.backbone
    bottleneck = model.bottleneck
    position_embed = model.pos_emb_z0
    torch_model = Backbone_Bottleneck_PE(backbone, bottleneck, position_embed)
    print(torch_model)
    # get the template
    img_z, mask_z = get_data(bs, z_sz, dtype=torch.float)
    # forward the template
    torch_outs = torch_model(img_z)
    torch.onnx.export(
        torch_model,  # model being run
        (img_z),  # model input (or a tuple for multiple inputs)
        save_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["img"],  # the model's input names
        output_names=[
            "feat_z",
            "mask_z",
            "pos_z",
        ],  # the model's output names
    )
