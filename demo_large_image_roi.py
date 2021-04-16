from mmdet.apis import init_detector, inference_detector, show_result, draw_poly_detections
import mmcv
from mmcv import Config
from mmdet.datasets import get_dataset
import cv2
import os
import numpy as np
from tqdm import tqdm
import DOTA_devkit.polyiou as polyiou
import math
import pdb
from extend import extend

import sys
sys.path.append(os.path.abspath(os.path.dirname("../aws_ec2_inference_server/")))
from utils.utils import get_gps_from_pix, is_path_exist

def py_cpu_nms_poly_fast_np(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    # scores = dets[:, 4]
    # print('dets[:, :]', dets[:, :], dets[:, :].shape)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou

        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class DetectorModel():
    def __init__(self,
                 config_file,
                 checkpoint_file):
        # init RoITransformer
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.cfg = Config.fromfile(self.config_file)
        self.data_test = self.cfg.data['test']
        self.dataset = get_dataset(self.data_test)
        self.classnames = self.dataset.CLASSES
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        self.extend = True
        self.extend_div_scale = 1/2


    def inference_single(self, imagname, slide_size, chip_size):
        img = mmcv.imread(imagname)
        
        if self.extend:
            try:
                img = extend(imagname, self.extend_div_scale)
                print("extended")
            except:
                self.extend = False

        height, width, channel = img.shape
        slide_h, slide_w = slide_size
        hn, wn = chip_size
        # TODO: check the corner case
        # import pdb; pdb.set_trace()

        total_detections = [np.zeros((0, 9)) for _ in range(len(self.classnames))]
        # total_detections = [np.zeros((0, 5)) for _ in range(len(self.classnames))]
        # print('total_detections: ', total_detections)

        for i in tqdm(range(int(width / slide_w + 1))):
            for j in range(int(height / slide_h) + 1):
                subimg = np.zeros((hn, wn, channel))
                # print('i: ', i, 'j: ', j)
                chip = img[j*slide_h:j*slide_h + hn, i*slide_w:i*slide_w + wn, :3]
                subimg[:chip.shape[0], :chip.shape[1], :] = chip

                chip_detections = inference_detector(self.model, subimg)
                # print('chip_detections: ', chip_detections)
                # print('result: ', result)
                for cls_id, name in enumerate(self.classnames):
                    chip_detections[cls_id][:, :8][:, ::2] = chip_detections[cls_id][:, :8][:, ::2] + i * slide_w
                    chip_detections[cls_id][:, :8][:, 1::2] = chip_detections[cls_id][:, :8][:, 1::2] + j * slide_h
                    # import pdb;pdb.set_trace()
                    # print("chip_detections.shape: ", chip_detections[cls_id].shape)
                    try:
                        total_detections[cls_id] = np.concatenate((total_detections[cls_id], chip_detections[cls_id]))
                    except Exception as e:
                        print("Error: ", e)
                        print(total_detections[cls_id].shape, chip_detections[cls_id].shape)
                        import pdb; pdb.set_trace()
        # nms
        for i in range(len(self.classnames)):
            keep = py_cpu_nms_poly_fast_np(total_detections[i], 0.1)
            total_detections[i] = total_detections[i][keep]
        return total_detections

    def det2json(dataset, result):
        for label in range(result):
            bboxes = results[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = self.classname[label]
        return data

    def inference_single_vis(self, srcpath, dstpath, slide_size, chip_size):
        detections = self.inference_single(srcpath, slide_size, chip_size)
        print('detections: ', detections)
        print('len(detections): ', len(detections))
        print('detections[0]: ', detections[0])
        print('len(detections[0]): ', len(detections[0]))
        print('self.classnames: ', self.classnames)
        img = draw_poly_detections(srcpath, detections, self.classnames, scale=1, threshold=0.3)
        cv2.imwrite(dstpath, img)

    def inference_image_list(self, moid, cropped_img_save_dir, cropped_image_list, inference_result_dictionary, logger, result_img_dir, model_path, label_dict,
                            slide_size=(512, 512), chip_size=(1024, 1024)):
        import time
        num_of_img = len(cropped_image_list)
        cnt = 0
        for image_name in cropped_image_list:
            img_path = os.path.join(cropped_img_save_dir, image_name)
            image = cv2.imread(img_path)
            img_height, img_width, _ = image.shape

            inference_start = time.time()

            '''resized_img, _, _, _, det_boxes_r_, det_scores_r_, det_category_r_ = \
                self.sess.run(
                    [self.img_batch, self.det_boxes_h, self.det_scores_h, self.det_category_h,
                     self.det_boxes_r, self.det_scores_r, self.det_category_r],
                    feed_dict={self.img_plac: image}
                )'''
            detections = self.inference_single(img_path, slide_size, chip_size)
            '''print('detections: ', detections)
            print('len(detections): ', len(detections))
            print('detections[0]: ', detections[0])
            print('len(detections[0]): ', len(detections[0]))
            print('self.classnames: ', self.classnames)'''

            inference_end = time.time()
            inference_time = inference_end - inference_start
            print('RoITransformer inference time : ', inference_time)

            image_latitude = float(image_name.split('_')[3][:-4])
            image_longitude = float(image_name.split('_')[2][:-4])
            
            threshold_dict = {
                        'flatroof': 0.05, 'solarpanel_flat': 0.05,
                        'solarpanel_slope': 0.05, 'parkinglot': 0.5,
                        'facility': 0.1, 'rooftop': 0.05,
                        'heliport_r': 0.05, 'heliport_h': 0.05
                        # 'heliport': 0.05
                    }

            '''threshold_dict = {
                'flatroof': 0.1, 'solarpanel_flat': 0.05,
                'solarpanel_slope': 0.05, 'parkinglot': 0.5,
                'facility': 0.3, 'rooftop': 0.05,
                'heliport_r': 0.05, 'heliport_h': 0.05
                #'heliport': 0.05
            }'''
            
            num_of_detected = 0
            num_confidence_over = 0
            detected_objects = []
            for j, name in enumerate(self.classnames):
                dets = detections[j]
                class_num = j
                # print('len(dets): ', len(dets), j, name)
                num_of_detected += len(dets)
                for det in dets:
                    scale = 1
                    # threshold = 0.3
                    threshold = threshold_dict[label_dict[class_num]]
                    zoom_level = 20
                    max_img_size = 640
                    
                    bbox = det[:8] * scale
                    score = det[-1]

                    if score < threshold:
                        continue
                    num_confidence_over += 1
                    confidence = score
                    bbox = list(map(int, bbox))
                    
                    if self.extend:
                        for i in range(0, len(bbox), 2):
                            bbox[i] = bbox[i] - int(self.extend_div_scale * img_width)
                        for i in range(1, len(bbox), 2):
                            bbox[i] = bbox[i] - int(self.extend_div_scale * img_height)
                    
                    detected_bbox = bbox
                    center_x = (bbox[0] + bbox[4])/2
                    center_y = (bbox[1] + bbox[5])/2
                    # bbox_info = np.array(det_boxes_r_[index]).astype(np.int64)
                    # x_c, y_c, w, h, theta = bbox_info[0], bbox_info[1], bbox_info[2], bbox_info[3], bbox_info[4]
                    # rotated_bbox_coord = np.int0(rect)
                    rotated_bbox_coord = [[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]]]
                    
                    (pointLng, pointLat) = get_gps_from_pix(image_latitude, image_longitude, zoom_level,
                                                            center_x, center_y, max_img_size, max_img_size)

                    detected_objects.append({'class': class_num, 'confidence': confidence,
                                             'latitude': pointLat, 'longitude': pointLng,
                                             'img_height': img_height, 'img_width': img_width,
                                             'bbox_center_x': center_x, 'bbox_center_y': center_y,
                                             'bbox_coord': rotated_bbox_coord, 'detected_bbox': detected_bbox})
                    

            inference_result_dictionary[moid][image_name] = detected_objects

            if_postprocess_iou = True
            offset = 10

            if if_postprocess_iou:
                result_dict = inference_result_dictionary[moid][image_name]
                group_of_nearer_box = []

                for idx_element, element in enumerate(result_dict):
                    if element == None:
                        continue
                    center_x, center_y = element['bbox_center_x'], element['bbox_center_y']
                    group = [element]
                    result_dict[idx_element] = None

                    for idx_near_element, near_element in enumerate(result_dict):
                        if near_element == None:
                            continue
                        near_center_x, near_center_y = near_element['bbox_center_x'], near_element['bbox_center_y']
                        if center_x - offset <= near_center_x and near_center_x <= center_x + offset:
                            if center_y - offset <= near_center_y and near_center_y <= center_y + offset:
                                group.append(near_element)
                                result_dict[idx_near_element] = None
                    group_of_nearer_box.append(group)

                for g_idx, group_ in enumerate(group_of_nearer_box):
                    best_conf = 0
                    best_idx = 0

                    if len(group_) > 1:
                        for idx, box in enumerate(group_):
                            conf = box['confidence']
                            if best_conf < conf:
                                best_conf = conf
                                best_idx = idx
                    group_of_nearer_box[g_idx] = group_[best_idx]

                inference_result_dictionary[moid][image_name] = group_of_nearer_box

            # For Development
            #####################################################################################
            IS_TEST=True
            model_name = str(model_path).split('/')[-1].split('.')[0]
            if IS_TEST:
                # Draw inference result image
                result_image_path = os.path.join(result_img_dir, model_name, moid)
                is_path_exist(result_image_path)
                '''result_image = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
                                                                      boxes=det_boxes_r_,
                                                                      labels=det_category_r_,
                                                                      scores=det_scores_r_)
                cv2.imwrite(result_image_path + '/' + image_name, result_image)'''
                img = draw_poly_detections(img_path, detections, self.classnames, scale=1, threshold=0.3, extend=self.extend, extend_div_scale=self.extend_div_scale)
                cv2.imwrite(result_image_path + '/' + image_name, img)

            #####################################################################################

            print("Predict result : ", image_name, inference_time)
            with open(result_image_path+'_inference_time.txt', 'a') as f:
                data = "{} {} {} {}\n".format(image_name, num_of_detected, num_confidence_over, inference_time)
                f.write(data)

            cnt += 1
            if cnt % 10 == 0:
                logger.info("Image inference [ %3d / %3d ]" % (cnt, num_of_img))
            elif cnt == num_of_img:
                logger.info("Image inference [ %3d / %3d ]" % (cnt, num_of_img))
        print("Inference finished. MOID number : ", moid)
        print('\n\n')
        return inference_result_dictionary


if __name__ == '__main__':
    roitransformer = DetectorModel(r'configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_dota_roof.py',
                  r'work_dirs/val_with_label_classes_8_cp1_val1_roitrans/epoch_12.pth')

    roitransformer.inference_single_vis(r'demo/[0,7](139.830825E,38.730867N)_center_(139.83039300000002E,38.731219N)min_(139.83125700000002E,38.730515000000004N)_max_zoom_20_size_640x640.png',  r'demo/roof2_roitrans_out.jpg',
                                        (512, 512),
                                       (1024, 1024))

