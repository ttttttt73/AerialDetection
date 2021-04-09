import warnings

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector
from extend import extend as extend_img

def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    img_transform = ImageTransform(
        size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, device)
    else:
        return _inference_generator(model, imgs, img_transform, device)


def _prepare_data(img, img_transform, cfg, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=cfg.data.test.img_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    return dict(img=[img], img_meta=[img_meta])


def _inference_single(model, img, img_transform, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, model.cfg, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def _inference_generator(model, imgs, img_transform, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, device)


# TODO: merge this method with the one in BaseDetector
def show_result(img, result, class_names, score_thr=0.3, out_file=None):
    """Visualize the detection results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)

def check_overlapping_parts(overlapping, text_rect, width, height):
    changed_text_rect = text_rect
    if overlapping is None:
        return changed_text_rect

    for start_point in overlapping:
        new_x = changed_text_rect[0]
        new_y = changed_text_rect[1]

        x = start_point[0]
        y = start_point[1]

        #절대 안겹칠 조건 : x가 x-width 보다 작거나 // x+width 보다 클 때
        # (continue)     y가 y-height 보다 작거나 // y+height 보다 클 때
        if (new_x < x-width) or (new_x > x+width):
            continue

        elif (new_y < y-height) or (new_y > y+height):
            continue

        else:
            # print("overlapped x, y")
            changed_text_rect = [new_x, new_y + height]
            changed_text_rect = check_overlapping_parts(overlapping, changed_text_rect, width, height)

    # print("result cnaged box : (" ,changed_text_rect[0], "," ,changed_text_rect[1], ")")
    return changed_text_rect


def draw_poly_detections(imgpath, detections, class_names, scale, threshold=0.2, extend=False):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb
    import cv2
    import random
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(imgpath)
    img_h, img_w = img.shape[:2]
    if extend:
        img = extend_img(imgpath)
    
    color_white = (255, 255, 255)

    drawed_counts = 0
    overlapping = []
    for j, name in enumerate(class_names):
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        threshold_dict = {
                        'flatroof': 0.05, 'solarpanel_flat': 0.05,
                        'solarpanel_slope': 0.05, 'parkinglot': 0.5,
                        'facility': 0.1, 'rooftop': 0.05,
                        'heliport': 0.05
                    }
        try:
            dets = detections[j]
            # print('len(dets): ', len(dets), j, name)
        except:
            pdb.set_trace()
        for det in dets:
            bbox = det[:8] * scale
            score = det[-1]
            # if score < threshold:
                # print('bbox: ', bbox, 'score: ', score, '--------> Score is lower than threshold')
            if score < threshold_dict[name]:
                print('bbox: ', bbox, 'score: ', score, '--------> Score is lower than threshold', name, threshold_dict[name])
                continue
            # print('bbox: ', bbox, 'score: ', score)
            # print('det: ', det)
            bbox = list(map(int, bbox))

            x_c = int((bbox[0] + bbox[4])/2)
            y_c = int((bbox[1] + bbox[5])/2)
            x_c_width = 150
            y_c_height = 15
            text_rect = [x_c, y_c]

            # 레이블을 찍을 위치에 이미 다른 레이블이 찍혀있다면 (위치가 중복된 경우) 위로 찍게 하기
            # 이미 찍은 점들과 위치를 비교해서 안겹치는 위치로 조정된 text_구역에 찍을 위치를 받아옴
            changed_text_rect = check_overlapping_parts(overlapping, text_rect, x_c_width, y_c_height)
            x_c = changed_text_rect[0]
            y_c = changed_text_rect[1]
            overlapping.append(changed_text_rect)

            # cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
                        
            '''cv2.line(img, (bbox[0], bbox[1]), (bbox[0], bbox[3]), color=color, thickness=2)
            cv2.line(img, (bbox[0], bbox[1]), (bbox[2], bbox[1]), color=color, thickness=2)
            cv2.line(img, (bbox[0], bbox[3]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.line(img, (bbox[2], bbox[3]), (bbox[2], bbox[1]), color=color, thickness=2)
            # cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2)
            cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)'''
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=(0, 255, 0), thickness=2)
                # print(i * 2, i * 2 + 1, (i+1) * 2, (i+1) * 2 + 1)
            '''for i in range(0, 8, 2):    
                cv2.putText(img, str(i+1), (bbox[i], bbox[i+1]), color=(0,255,250), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)'''
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=(0, 255, 0), thickness=2)
            cv2.rectangle(img, pt1=(x_c, y_c), 
                                pt2=(x_c + x_c_width, y_c + y_c_height), 
                                color=(0, 255, 0), 
                                thickness=-1)
            # cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
            #             color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
            cv2.putText(img, '%s %.3f' % (class_names[j], score), org=(x_c, y_c+10),
                         color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, thickness=2)
            drawed_counts += 1

    print("drawed counts: ", drawed_counts)
    cv2.putText(img, text=str(drawed_counts), org=((img.shape[1]) // 2, (img.shape[0]) // 2), fontFace=3, fontScale=1, color=(255, 0, 0), thickness=2)

    if extend:
        img = img[int(img_h/2):int(img_h/2)+int(img_h), int(img_w/2):int(img_w/2)+int(img_w)]

    return img

