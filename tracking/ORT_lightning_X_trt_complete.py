import argparse
import torch
import _init_paths
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_stark_lightning_x_trt
from lib.config.stark_lightning_X_trt.config import cfg, update_config_from_file
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn as nn
import torch.nn.functional as F

# for onnx conversion and inference
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


def get_data(bs=1, sz_x=256, hw_z=64, c=256, dtype=torch.float16):
    img_x = torch.randn(bs, 3, sz_x, sz_x, requires_grad=True, dtype=dtype)
    mask_x = torch.rand(sz_x, sz_x, requires_grad=True, dtype=dtype) > 0.5
    feat_vec_z = torch.randn(hw_z, bs, c, requires_grad=True, dtype=dtype)  # HWxBxC
    mask_vec_z = torch.rand(bs, hw_z, requires_grad=True, dtype=dtype) > 0.5  # BxHW
    pos_vec_z = torch.randn(hw_z, bs, c, requires_grad=True, dtype=dtype)  # HWxBxC
    return img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z


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
    def __init__(self, size=320):
        super().__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.size = size

    def forward(self, in_img: torch.Tensor) -> torch.Tensor:
        in_img = in_img.to(torch.float)

        mask = in_img.mean(dim=1) > 0
        mask = mask.to(torch.float) * 255
        mask = mask.unsqueeze(0)
        mask = self.max_pool2d(mask)
        mask = torch.clamp(mask, 0, 1)
        mask = mask.squeeze(0).squeeze(0)
        mask = mask.to(torch.bool)
        mask = torch.bitwise_not(mask)

        return mask


class STARK(nn.Module):
    def __init__(self, backbone, bottleneck, position_embed, transformer, box_head):
        super(STARK, self).__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.position_embed = position_embed
        self.transformer = transformer
        self.box_head = box_head
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz**2)
        self.preprocessor = Preprocessor()
        self.mask_nn = MaskModel()

    def forward(
        self,
        img: torch.Tensor,
        feat_z: torch.Tensor,
        mask_z: torch.Tensor,
        pos_z: torch.Tensor,
    ):
        mask = self.mask_nn(img)
        img, mask = self.preprocessor(img, mask)

        # run the backbone
        feat = self.bottleneck(self.backbone(img))  # BxCxHxW
        mask_down = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(
            torch.bool
        )[0]
        pos_embed = self.position_embed(
            bs=1
        )  # 1 is the batch-size. output size is BxCxHxW
        # adjust shape
        feat_vec_x = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_vec_x = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec_x = mask_down.flatten(1)  # BxHW
        # concat with the template-related results
        feat_vec = torch.cat([feat_z, feat_vec_x], dim=0)
        mask_vec = torch.cat([mask_z, mask_vec_x], dim=1)
        pos_vec = torch.cat([pos_z, pos_vec_x], dim=0)
        # get q, k, v
        q = feat_vec_x + pos_vec_x
        k = feat_vec + pos_vec
        v = feat_vec
        key_padding_mask = mask_vec
        # run the transformer encoder
        memory = self.transformer(q, k, v, key_padding_mask=key_padding_mask)
        fx = memory[-self.feat_len_s :].permute(1, 2, 0).contiguous()  # (B, C, H_x*W_x)
        fx_t = fx.view(
            *fx.shape[:2], self.feat_sz_s, self.feat_sz_s
        ).contiguous()  # fx tensor 4D (B, C, H_x, W_x)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(fx_t))
        return outputs_coord


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


if __name__ == "__main__":
    load_checkpoint = True
    save_name = "complete.onnx"
    # update cfg
    args = parse_args()
    yaml_fname = "experiments/%s/%s.yaml" % (args.script, args.config)
    update_config_from_file(yaml_fname)
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
    position_embed = model.pos_emb_x
    transformer = model.transformer
    box_head = model.box_head
    box_head.coord_x = box_head.coord_x.cpu()
    box_head.coord_y = box_head.coord_y.cpu()
    torch_model = STARK(backbone, bottleneck, position_embed, transformer, box_head)
    print(torch_model)
    torch.save(torch_model.state_dict(), "complete.pth")
    # get the network input
    bs = 1
    sz_x = cfg.TEST.SEARCH_SIZE
    hw_z = cfg.DATA.TEMPLATE.FEAT_SIZE**2
    c = cfg.MODEL.HIDDEN_DIM
    print(bs, sz_x, hw_z, c)
    img_x, mask_x, feat_vec_z, mask_vec_z, pos_vec_z = get_data(
        bs=bs, sz_x=sz_x, hw_z=hw_z, c=c, dtype=torch.float
    )

    torch_outs = torch_model(img_x, feat_vec_z, mask_vec_z, pos_vec_z)
    torch.onnx.export(
        torch_model,  # model being run
        (
            img_x,
            feat_vec_z,
            mask_vec_z,
            pos_vec_z,
        ),  # model input (a tuple for multiple inputs)
        save_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[
            "img_x",
            "feat_z",
            "mask_z",
            "pos_z",
        ],  # model's input names
        output_names=["outputs_coord"],  # the model's output names
    )
