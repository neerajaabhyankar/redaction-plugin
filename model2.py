import os
import time
import argparse
import math
from itertools import product
from typing import Tuple

import torch
import numpy as np
import cv2

from yakhyo_tinyface import RFB
from yakhyo_tinyface.box_utils import decode, decode_landmarks, nms


WEIGHTS_PATH = './yakhyo_tinyface/weights/'

cfg_mnet = {
    'name': 'retinaface',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 64,
    'out_channel': 64
}

cfg_slim = {
    'name': 'slim',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
}

cfg_rfb = {
    'name': 'rfb',
    'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
}

def get_config(network):
    configs = {
        "retinaface": cfg_mnet,
        "slim": cfg_slim,
        "rfb": cfg_rfb,
    }
    return configs.get(network, None)

VIS_THRESHOLD = 0.6
PRE_NMS_TOPK = 5000
NMS_THRESHOLD = 0.4
POST_NMS_TOPK = 750
CONF_THRESHOLD = 0.02

def draw_detections(original_image, detections):
    """
    Draws bounding boxes and landmarks on the image based on multiple detections.

    Args:
        original_image (ndarray): The image on which to draw detections.
        detections (ndarray): Array of detected bounding boxes and landmarks.
    """

    # Colors for visualization
    LANDMARK_COLORS = [
        (0, 0, 255),    # Right eye (Red)
        (0, 255, 255),  # Left eye (Yellow)
        (255, 0, 255),  # Nose (Magenta)
        (0, 255, 0),    # Right mouth (Green)
        (255, 0, 0)     # Left mouth (Blue)
    ]
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)

    print(f"#faces: {len(detections)}")

    # Slice arrays efficiently
    boxes = detections[:, 0:4].astype(np.int32)
    scores = detections[:, 4]
    landmarks = detections[:, 5:15].reshape(-1, 5, 2).astype(np.int32)

    for box, score, landmark in zip(boxes, scores, landmarks):
        # Draw bounding box
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), BOX_COLOR, 2)

        # Draw confidence score
        text = f"{score:.2f}"
        cx, cy = box[0], box[1] + 12
        cv2.putText(original_image, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_COLOR)

        # Draw landmarks
        for point, color in zip(landmark, LANDMARK_COLORS):
            cv2.circle(original_image, point, 1, color, 4)


class PriorBox:
    def __init__(self, cfg: dict, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.clip = cfg['clip']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.feature_maps = [[
            math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)] for step in self.steps
        ]

    def generate_anchors(self) -> torch.Tensor:
        """Generate anchor boxes based on configuration and image size"""
        anchors = []
        for k, (map_height, map_width) in enumerate(self.feature_maps):
            step = self.steps[k]
            for i, j in product(range(map_height), range(map_width)):
                for min_size in self.min_sizes[k]:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]

                    dense_cx = [x * step / self.image_size[1] for x in [j+0.5]]
                    dense_cy = [y * step / self.image_size[0] for y in [i+0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference Arguments for Face Detection")

    # Model and device options
    parser.add_argument(
        '--network',
        type=str,
        default='rfb',
        choices=['retinaface', 'slim', 'rfb'],
        help='Select a model architecture for face detection'
    )

    # Detection settings
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=CONF_THRESHOLD,
        help='Confidence threshold for filtering detections'
    )
    parser.add_argument(
        '--pre-nms-topk',
        type=int,
        default=PRE_NMS_TOPK,
        help='Maximum number of detections to consider before applying NMS'
    )
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=NMS_THRESHOLD,
        help='Non-Maximum Suppression (NMS) threshold'
    )
    parser.add_argument(
        '--post-nms-topk',
        type=int,
        default=POST_NMS_TOPK,
        help='Number of highest scoring detections to keep after NMS'
    )

    # Output options
    parser.add_argument(
        '-s', '--save-image',
        action='store_true',
        help='Save the detection results as images'
    )
    parser.add_argument(
        '-v', '--vis-threshold',
        type=float,
        default=VIS_THRESHOLD,
        help='Visualization threshold for displaying detections'
    )

    # Image input
    parser.add_argument(
        '--image-path',
        type=str,
        default='./test_frame.jpg',
        help='Path to the input image'
    )

    return parser.parse_args()


@torch.no_grad()
def inference(model, image):
    model.eval()
    loc, conf, landmarks = model(image)

    loc = loc.squeeze(0)
    conf = conf.squeeze(0)
    landmarks = landmarks.squeeze(0)

    return loc, conf, landmarks

import fiftyone as fo

def apply_model_adhoc(samples: fo.DatasetView):

    # load configuration and device setup
    cfg = cfg_rfb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFB(cfg=cfg)
    state_dict = torch.load(WEIGHTS_PATH + 'rfb.pth', map_location=device, weights_only=True)
    
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    model.to(device)
    model.eval()

    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    for si, sample in enumerate(samples):

        # read image
        original_image = cv2.imread(sample.filepath, cv2.IMREAD_COLOR)
        image = np.float32(original_image)
        img_height, img_width, _ = image.shape

        # normalize image
        image -= rgb_mean
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).unsqueeze(0)  # 1CHW
        image = image.to(device)

        # forward pass
        loc, conf, landmarks = inference(model, image)
        
        # generate anchor boxes
        priorbox = PriorBox(cfg, image_size=(img_height, img_width))
        priors = priorbox.generate_anchors().to(device)

        # decode boxes and landmarks
        boxes = decode(loc, priors, cfg['variance'])
        landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

        # scale adjustments
        bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
        boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

        landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
        landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

        scores = conf.cpu().numpy()[:, 1]

        # filter by confidence threshold
        inds = scores > CONF_THRESHOLD
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        # sort by scores
        order = scores.argsort()[::-1][:PRE_NMS_TOPK]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # apply NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, NMS_THRESHOLD)

        detections = detections[keep]
        landmarks = landmarks[keep]

        # keep top-k detections and landmarks
        detections = detections[:POST_NMS_TOPK]
        landmarks = landmarks[:POST_NMS_TOPK]

        # concatenate detections and landmarks
        detections = np.concatenate((detections, landmarks), axis=1)

        # show image
        detections = detections[detections[:, 4] >= VIS_THRESHOLD]

        # Convert to [top-left-x, top-left-y, width, height] in normalized coordinates (0-1)
        boxes_xyxy = detections[:, :4]  # [xmin, ymin, xmax, ymax] in pixel coordinates
        
        # Convert to [x, y, width, height] format where x,y is top-left
        boxes_xywh = np.zeros_like(boxes_xyxy)
        boxes_xywh[:, 0] = boxes_xyxy[:, 0]  # x (top-left)
        boxes_xywh[:, 1] = boxes_xyxy[:, 1]  # y (top-left)
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # width
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # height
        
        # Normalize coordinates by image dimensions
        boxes_xywh[:, 0] /= img_width   # normalize x
        boxes_xywh[:, 1] /= img_height  # normalize y
        boxes_xywh[:, 2] /= img_width   # normalize width
        boxes_xywh[:, 3] /= img_height   # normalize height
        
        # Create fo.Detection objects with confidence scores
        fo_detections = [
            fo.Detection(
                label="face",
                bounding_box=box.tolist(),
                confidence=float(score)
            )
            for box, score in zip(boxes_xywh, detections[:, 4])
        ]

        detections_object = fo.Detections(
            detections=fo_detections,
            field="predictions-yakhyo-tinyface"
        )
        sample.set_field("predictions-yakhyo-tinyface", detections_object)
        sample.save()

    return samples


def main(params):
    # load configuration and device setup
    cfg = get_config(params.network)
    if cfg is None:
        raise KeyError(f"Config file for {params.network} not found!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_mean = (104, 117, 123)
    resize_factor = 1

    # model initialization
    if params.network == "retinaface":
        # model = RetinaFace(cfg=cfg)
        model = None
        state_dict = None
        print("not downloaded")
    elif params.network == "slim":
        # model = SlimFace(cfg=cfg)
        model = None
        state_dict = None
        print("not downloaded")
    elif params.network == "rfb":
        model = RFB(cfg=cfg)
        state_dict = torch.load(WEIGHTS_PATH + 'rfb.pth', map_location=device, weights_only=True)
    else:
        raise NameError("Please choose existing face detection method!")

    model.to(device)
    model.eval()

    # loading state_dict
    
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")

    # read image
    original_image = cv2.imread(params.image_path, cv2.IMREAD_COLOR)
    image = np.float32(original_image)
    img_height, img_width, _ = image.shape

    # normalize image
    image -= rgb_mean
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0)  # 1CHW
    image = image.to(device)

    # forward pass
    loc, conf, landmarks = inference(model, image)

    # generate anchor boxes
    priorbox = PriorBox(cfg, image_size=(img_height, img_width))
    priors = priorbox.generate_anchors().to(device)

    # decode boxes and landmarks
    boxes = decode(loc, priors, cfg['variance'])
    landmarks = decode_landmarks(landmarks, priors, cfg['variance'])

    # scale adjustments
    bbox_scale = torch.tensor([img_width, img_height] * 2, device=device)
    boxes = (boxes * bbox_scale / resize_factor).cpu().numpy()

    landmark_scale = torch.tensor([img_width, img_height] * 5, device=device)
    landmarks = (landmarks * landmark_scale / resize_factor).cpu().numpy()

    scores = conf.cpu().numpy()[:, 1]

    # filter by confidence threshold
    inds = scores > params.conf_threshold
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    scores = scores[inds]

    # sort by scores
    order = scores.argsort()[::-1][:params.pre_nms_topk]
    boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

    # apply NMS
    detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(detections, params.nms_threshold)

    detections = detections[keep]
    landmarks = landmarks[keep]

    # keep top-k detections and landmarks
    detections = detections[:params.post_nms_topk]
    landmarks = landmarks[:params.post_nms_topk]

    # concatenate detections and landmarks
    detections = np.concatenate((detections, landmarks), axis=1)

    # show image
    detections = detections[detections[:, 4] >= params.vis_threshold]
    draw_detections(original_image, detections)
    # save image
    im_name = os.path.splitext(os.path.basename(params.image_path))[0]
    save_name = f"{im_name}_{params.network}_out.jpg"
    cv2.imwrite(save_name, original_image)
    print(f"Image saved at '{save_name}'")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)