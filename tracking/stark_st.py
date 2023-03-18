from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
import importlib

# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.stark import build_starkst
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box


class STARK_ST(BaseTracker):
    def __init__(self, params=None):
        super(STARK_ST, self).__init__(params)

        # yaml_fname = "/home/nik/Projects/Diplomska/stark-on-depthai/Stark/experiments/stark_st2/baseline_R101.yaml"
        yaml_fname = "/home/nik/Projects/Diplomska/stark-on-depthai/Stark/experiments/stark_st2/baseline.yaml"
        config_module = importlib.import_module(f"lib.config.stark_st2.config")
        cfg = config_module.cfg
        config_module.update_config_from_file(yaml_fname)

        network = build_starkst(cfg)
        network.load_state_dict(
            torch.load(
                # "/home/nik/Projects/Diplomska/stark-on-depthai/Stark/checkpoints/train/stark_st/baseline_R101/STARKST_ep0050.pth.tar",
                "/home/nik/Projects/Diplomska/stark-on-depthai/Stark/checkpoints/train/stark_st/baseline_long/STARKST_ep0100.pth.tar",
                map_location="cpu",
            )["net"],
            strict=True,
        )
        self.cfg = cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        self.update_intervals = [10]  # self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

    def initialize(self, image, info: dict):
        # initialize z_dict_list
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(
            image,
            info["init_bbox"],
            2.0,
            output_sz=128,
        )
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        # save states
        self.state = info["init_bbox"]
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            5.0,
            output_sz=320,
        )  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(
                seq_dict=seq_dict, run_box_head=True, run_cls_head=True
            )
        # get the final result
        pred_boxes = out_dict["pred_boxes"].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (
            pred_boxes.mean(dim=0) * 320 / resize_factor
        ).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(
            self.map_box_back(pred_box, resize_factor), H, W, margin=10
        )
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        # update template
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.5:
                z_patch_arr, _, z_amask_arr = sample_target(
                    image,
                    self.state,
                    2.0,
                    output_sz=128,
                )  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                self.z_dict_list[
                    idx + 1
                ] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        return {"target_bbox": self.state, "conf_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = (
            self.state[0] + 0.5 * self.state[2],
            self.state[1] + 0.5 * self.state[3],
        )
        cx, cy, w, h = pred_box
        half_side = 0.5 * 320 / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = (
            self.state[0] + 0.5 * self.state[2],
            self.state[1] + 0.5 * self.state[3],
        )
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * 320 / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return STARK_ST


def build_stark_st():
    print("AAAAAAAAAAAAAAAAA")
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    # args = parse_args()
    """update cfg"""
    yaml_fname = "/home/nik/Projects/Diplomska/stark-on-depthai/Stark/experiments/stark_st2/baseline_R101.yaml"
    config_module = importlib.import_module(f"lib.config.stark_st2.config")
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    """set some values"""
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    h_dim = cfg.MODEL.HIDDEN_DIM
    """import stark network module"""
    model_module = importlib.import_module("lib.models.stark")
    model_constructor = model_module.build_starkst
    model = model_constructor(cfg)
    # transfer to device
    model = model.to(device)
    return model
