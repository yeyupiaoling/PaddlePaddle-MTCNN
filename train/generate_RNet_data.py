import os
import pickle
import random
import shutil

import paddle.fluid as fluid
import cv2
import numpy as np
from data_format_converter import convert_data
from tqdm import tqdm
from utils import IOU, crop_landmark_image, combine_data_list, read_annotation, convert_to_square

np.set_printoptions(threshold=np.inf)

# 获取执行器
# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)

# 从保存的模型文件中获取预测程序、输入数据的名称和输出层
[infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='../infer_model/PNet',
                                                                               executor=exe)


def predict(infer_data):
    # 添加待预测的图片
    infer_data = infer_data[np.newaxis,]
    # 执行预测
    cls_prob, bbox_pred, landmark_pred = exe.run(program=infer_program,
                                                 feed={feeded_var_names[0]: infer_data},
                                                 fetch_list=target_vars)
    return cls_prob, bbox_pred


def processed_image(img, scale):
    '''预处理数据，转化图像尺度并对像素归一到[-1,1]
    '''
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    # 把图片转换成numpy值
    image = np.array(img_resized).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 转换成BGR
    image = (image[(2, 1, 0), :, :] - 127.5) / 128
    return image


def py_nms(dets, thresh):
    '''剔除太相似的box'''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def generate_bbox(cls_map, reg, scale, threshold):
    """
     得到对应原图的box坐标，分类分数，box偏移量
    """
    # pnet大致将图像size缩小2倍
    stride = 2

    cellsize = 12

    # 将置信度高的留下
    t_index = np.where(cls_map > threshold)

    # 没有人脸
    if t_index[0].size == 0:
        return np.array([])
    # 偏移量
    dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    # 对应原图的box坐标，分类分数，box偏移量
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape[n,9]
    return boundingbox.T


def detect_pnet(im, min_face_size, scale_factor, thresh):
    '''通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    '''
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict(im_resized)
        cls_cls_map = cls_cls_map.transpose((1, 2, 0))
        reg = reg.transpose((1, 2, 0))
        boxes = generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, thresh)
        print(boxes.shape)
        current_scale *= scale_factor  # 继续缩小图像做金字塔
        im_resized = processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        if boxes.size == 0:
            continue
        # 非极大值抑制留下重复低的box
        keep = py_nms(boxes[:, :5], 0.5)
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if len(all_boxes) == 0:
        return None, None, None
    all_boxes = np.vstack(all_boxes)
    # 将金字塔之后的box也进行非极大值抑制
    keep = py_nms(all_boxes[:, 0:5], 0.7)
    all_boxes = all_boxes[keep]
    boxes = all_boxes[:, :5]
    # box的长宽
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
    # 对应原图的box坐标和分数
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                         all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                         all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                         all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                         all_boxes[:, 4]])
    boxes_c = boxes_c.T

    return boxes, boxes_c, None


def save_hard_example(save_dir, save_size, data, neg_dir, pos_dir, part_dir):
    '''将网络识别的box用来裁剪原图像作为下一个网络的输入'''

    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)

    # 保存标注数据的文件
    pos_file = open(os.path.join(save_dir, 'positive.txt'), 'w')
    neg_file = open(os.path.join(save_dir, 'negative.txt'), 'w')
    part_file = open(os.path.join(save_dir, 'part.txt'), 'w')

    # 读取识别结果
    det_boxes = pickle.load(open(os.path.join(save_dir, 'detections.pkl'), 'rb'))

    assert len(det_boxes) == num_of_images, "弄错了"

    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    for im_idx, dets, gts in tqdm(zip(im_idx_list, det_boxes, gt_boxes_list)):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # 转换成正方形
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # 除去过小的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IOU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (save_size, save_size),
                                    interpolation=cv2.INTER_LINEAR)

            # 划分种类
            if np.max(Iou) < 0.3 and neg_num < 60:

                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)

                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:

                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # 偏移量
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # pos和part
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


# 截取pos,neg,part三种类型图片并resize成24x24大小作为RNet的输入
def crop_24_box_image(data_path, base_dir, filename, min_face_size, scale_factor, thresh):
    # pos，part,neg裁剪图片放置位置
    pos_save_dir = os.path.join(data_path, '24/positive')
    part_save_dir = os.path.join(data_path, '24/part')
    neg_save_dir = os.path.join(data_path, '24/negative')
    # RNet数据地址
    save_dir = os.path.join(data_path, '24/')

    # 创建文件夹
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)

    # 读取标注数据
    data = read_annotation(base_dir, filename)
    all_boxes = []
    landmarks = []
    empty_array = np.array([])

    print("开始识别")
    # 使用PNet模型识别图片
    for image_path in tqdm(data['images']):
        im = cv2.imread(image_path)
        boxes, boxes_c, _ = detect_pnet(im, min_face_size, scale_factor, thresh)
        if boxes_c is None:
            all_boxes.append(empty_array)
            landmarks.append(empty_array)
            continue

    # 把识别结果存放在文件中
    save_file = os.path.join(save_dir, 'detections.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(all_boxes, f, 1)

    print('开始生成图像')
    save_hard_example(save_dir, 24, data, neg_save_dir, pos_save_dir, part_save_dir)


# 合并图像后删除原来的文件
def delete_old_img(old_image_folder):
    shutil.rmtree(os.path.join(old_image_folder, '24', 'positive'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, '24', 'negative'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, '24', 'part'), ignore_errors=True)
    shutil.rmtree(os.path.join(old_image_folder, '24', 'landmark'), ignore_errors=True)


if __name__ == '__main__':
    data_path = '../data/'
    base_dir = '../data/WIDER_train/'
    filename = '../data/wider_face_train_bbx_gt.txt'
    min_face_size = 20
    scale_factor = 0.79
    thresh = 0.6
    # 获取人脸的box图片数据
    # print('开始生成bbox图像数据')
    # crop_24_box_image(data_path, base_dir, filename, min_face_size, scale_factor, thresh)
    # # 获取人脸关键点的数据
    # print('开始生成landmark图像数据')
    # crop_landmark_image(data_path, 24, argument=True)
    # # 合并数据列表
    # print('开始合成数据列表')
    # combine_data_list(os.path.join(data_path, '24'))
    # # 合并图像数据
    # print('开始合成图像文件')
    # convert_data(os.path.join(data_path, '12'), os.path.join(data_path, '12', 'all_data'))
    # # 删除旧数据
    # print('开始删除就得图像文件')
    # delete_old_img(data_path)

    im = cv2.imread('222.jpg')
    net_size = 12
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    cls_prob, bbox_pred = predict(im_resized)
    print(cls_prob.shape)
    print(bbox_pred.shape)


