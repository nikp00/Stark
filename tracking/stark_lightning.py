import torch
from torch import nn
import os
import sys
from torch.nn import functional as F
import numpy as np
import math
import cv2

# import _init_paths
from lib.config.stark_lightning_X_trt.config import cfg, update_config_from_file
from lib.models.stark.repvgg import repvgg_model_convert
from lib.models.stark import build_stark_lightning_x_trt
from lib.test.evaluation.environment import env_settings
from lib.utils.box_ops import box_xyxy_to_cxcywh


from typing import Tuple, Union

sys.path.insert(0, "/home/nik/Projects/Diplomska/stark-on-depthai/Stark")


class StarkLightningTracker:
    TEMPLATE_SIZE = 128
    TEMPLATE_FACTOR = 2.0
    SEARCH_SIZE = 320
    SEARCH_FACTOR = 5.0

    def __init__(self) -> None:
        yaml_frame = "/home/nik/Projects/Diplomska/stark-on-depthai/Stark/experiments/stark_lightning_X_trt/baseline_rephead_4_lite_search5.yaml"
        update_config_from_file(yaml_frame)
        print(cfg)
        print("done")
        model = build_stark_lightning_x_trt(cfg, phase="test")
        save_dir = env_settings().save_dir
        save_dir = env_settings().save_dir
        checkpoint_name = os.path.join(
            save_dir,
            "/home/nik/Projects/Diplomska/stark-on-depthai/Stark/checkpoints/train/stark_lightning_X_trt/baseline_rephead_4_lite_search5/STARKLightningXtrt_ep0500.pth.tar",
        )
        model.load_state_dict(
            torch.load(checkpoint_name, map_location="cpu")["net"], strict=True
        )

        model = repvgg_model_convert(model)
        model.eval()

        self.backbone = Backbone_Bottleneck_PE(
            model.backbone, model.bottleneck, model.pos_emb_z0
        )
        self.backbone.cuda(0)

        self.complete = STARK(
            model.backbone,
            model.bottleneck,
            model.pos_emb_x,
            model.transformer,
            model.box_head,
        )
        self.complete.cuda(0)
        self.complete.box_head.coord_x.cuda(0)
        self.complete.box_head.coord_y.cuda(0)

        self.feat_z = None
        self.mask_z = None
        self.pos_z = None
        self.state = None

    def sample_target(
        self,
        im: np.ndarray,
        bbox: list,
        search_area_factor: float,
        output_sz: Union[int, float],
    ):
        if not isinstance(bbox, list):
            x, y, w, h = bbox.tolist()
        else:
            x, y, w, h = bbox
        x, y, w, h = bbox

        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception("Too small bounding box.")

        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz

        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad : y2 - y2_pad, x1 + x1_pad : x2 - x2_pad, :]

        # Pad
        im_crop_padded = cv2.copyMakeBorder(
            im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT
        )

        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        return im_crop_padded, resize_factor

    def map_box_back(
        self,
        pred_box: np.ndarray,
        state: np.ndarray,
        resize_factor: float,
        search_size: float,
    ) -> np.ndarray:
        cx_prev, cy_prev = (
            state[0] + 0.5 * state[2],
            state[1] + 0.5 * state[3],
        )
        cx, cy, w, h = pred_box
        half_side = 0.5 * search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return np.array([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h])

    def clip_box(self, box: np.ndarray, H: int, W: int, margin: int = 0) -> np.ndarray:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W - margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H - margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2 - x1)
        h = max(margin, y2 - y1)
        return np.array([x1, y1, w, h])

    def initialize(
        self, input_frame: np.ndarray, bbox: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        img, _ = self.sample_target(
            input_frame, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE
        )

        img = img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img = img.astype(np.float32)

        self.feat_z, self.mask_z, self.pos_z = self.backbone(
            torch.from_numpy(img).cuda(0)
        )

        self.feat_z = self.feat_z.detach().cpu().numpy()
        self.mask_z = self.mask_z.detach().cpu().numpy()
        self.pos_z = self.pos_z.detach().cpu().numpy()

        self.state = np.array(bbox)

        return input_frame, self.state.tolist()

    def track(
        self,
        input_frame: np.ndarray,
    ):
        H, W, _ = input_frame.shape
        img_x, resize_factor = self.sample_target(
            input_frame, self.state, self.SEARCH_FACTOR, self.SEARCH_SIZE
        )  # (x1, y1, w, h)

        img_x = img_x[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_x = img_x.astype(np.float32)

        outputs_coord = self.complete(
            torch.from_numpy(img_x).cuda(0),
            torch.from_numpy(self.feat_z).cuda(0),
            torch.from_numpy(self.mask_z).cuda(0),
            torch.from_numpy(self.pos_z).cuda(0),
        )
        outputs_coord = outputs_coord.detach().cpu().numpy()[0]

        print(outputs_coord)

        pred_box = outputs_coord * self.SEARCH_SIZE / resize_factor
        self.state = self.clip_box(
            self.map_box_back(pred_box, self.state, resize_factor, self.SEARCH_SIZE),
            H,
            W,
            margin=10,
        )

        return input_frame, self.state.tolist()


class Backbone_Bottleneck_PE(nn.Module):
    def __init__(self, backbone, bottleneck, position_embed):
        super(Backbone_Bottleneck_PE, self).__init__()
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.position_embed = position_embed
        self.preprocessor = Preprocessor()
        self.mask_nn = MaskModel(128)

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
        self.mask_nn = MaskModel(320)

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


class Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = (
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
            .reshape((1, 3, 1, 1))
            .cuda(0)
        )
        self.std = (
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
            .reshape((1, 3, 1, 1))
            .cuda(0)
        )

    def forward(
        self, patch: torch.Tensor, amask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_4d = (patch / 255.0 - self.mean) / self.std  # (1, 3, H, W)

        # Deal with the attention mask
        amask_3d = amask.unsqueeze(0)  # (1,H,W)
        return patch_4d, amask_3d.to(torch.bool)


if __name__ == "__main__":
    model = StarkLightningTracker()
