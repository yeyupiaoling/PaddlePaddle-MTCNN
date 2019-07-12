import cv2
import paddle.fluid as fluid
import numpy as np
from utils import *

# 获取执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
pnet_exe = fluid.Executor(place)
rnet_exe = fluid.Executor(place)
onet_exe = fluid.Executor(place)

infer_pnet_scope = fluid.core.Scope()
infer_rnet_scope = fluid.core.Scope()
infer_onet_scope = fluid.core.Scope()


def predict_pnet(infer_data):
    with fluid.scope_guard(infer_pnet_scope):
        # 从保存的模型文件中获取预测程序、输入数据的名称和输出层
        [infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='../infer_model/PNet',
                                                                                       executor=pnet_exe)
        # 添加待预测的图片
        infer_data = infer_data[np.newaxis,]
        # 执行预测
        cls_prob, bbox_pred, landmark_pred = pnet_exe.run(program=infer_program,
                                                          feed={feeded_var_names[0]: infer_data},
                                                          fetch_list=target_vars)
        return cls_prob, bbox_pred


def predict_rnet(infer_data):
    with fluid.scope_guard(infer_rnet_scope):
        # 从保存的模型文件中获取预测程序、输入数据的名称和输出层
        [infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='../infer_model/RNet',
                                                                                       executor=rnet_exe)
        # 执行预测
        cls_prob, bbox_pred, landmark_pred = rnet_exe.run(program=infer_program,
                                                          feed={feeded_var_names[0]: infer_data},
                                                          fetch_list=target_vars)
        return cls_prob, bbox_pred


def predict_onet(infer_data):
    with fluid.scope_guard(infer_onet_scope):
        # 从保存的模型文件中获取预测程序、输入数据的名称和输出层
        [infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='../infer_model/ONet',
                                                                                       executor=onet_exe)
        # 执行预测
        cls_prob, bbox_pred, landmark_pred = onet_exe.run(program=infer_program,
                                                          feed={feeded_var_names[0]: infer_data},
                                                          fetch_list=target_vars)
        return cls_prob, bbox_pred, landmark_pred


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


def detect_pnet(im, min_face_size, scale_factor, thresh):
    """通过pnet筛选box和landmark
    参数：
      im:输入图像[h,2,3]
    """
    net_size = 12
    # 人脸和输入图像的比率
    current_scale = float(net_size) / min_face_size
    im_resized = processed_image(im, current_scale)
    _, current_height, current_width = im_resized.shape
    all_boxes = list()
    # 图像金字塔
    while min(current_height, current_width) > net_size:
        # 类别和box
        cls_cls_map, reg = predict_pnet(im_resized)
        cls_cls_map = cls_cls_map.transpose((1, 2, 0))
        reg = reg.transpose((1, 2, 0))
        boxes = generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, thresh)
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

    return boxes_c


def detect_rnet(im, dets, thresh):
    """通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
    """
    h, w, c = im.shape
    # 将pnet的box变成包含它的正方形，可以避免信息损失
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    # 调整超出图像的box
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    delete_size = np.ones_like(tmpw) * 20
    ones = np.ones_like(tmpw)
    zeros = np.zeros_like(tmpw)
    num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
    cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
    for i in range(num_boxes):
        # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
        if tmph[i] < 20 or tmpw[i] < 20:
            continue
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        try:
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
        except:
            continue
    cropped_ims = cropped_ims.transpose((0, 3, 1, 2))
    cls_scores, reg = predict_rnet(cropped_ims)
    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
    else:
        return None

    keep = py_nms(boxes, 0.6)
    boxes = boxes[keep]
    # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
    boxes_c = calibrate_box(boxes, reg[keep])
    return boxes_c


def detect_onet(im, dets, thresh):
    """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
    h, w, c = im.shape
    dets = convert_to_square(dets)
    dets[:, 0:4] = np.round(dets[:, 0:4])
    [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
    num_boxes = dets.shape[0]
    cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
    for i in range(num_boxes):
        tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
        tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
        cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

    cropped_ims = cropped_ims.transpose((0, 3, 1, 2))
    cls_scores, reg, landmark = predict_onet(cropped_ims)

    cls_scores = cls_scores[:, 1]
    keep_inds = np.where(cls_scores > thresh)[0]
    if len(keep_inds) > 0:
        boxes = dets[keep_inds]
        boxes[:, 4] = cls_scores[keep_inds]
        reg = reg[keep_inds]
        landmark = landmark[keep_inds]
    else:
        return None, None

    w = boxes[:, 2] - boxes[:, 0] + 1

    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    boxes_c = calibrate_box(boxes, reg)

    keep = py_nms(boxes_c, 0.6)
    boxes_c = boxes_c[keep]
    landmark = landmark[keep]
    return boxes_c, landmark


def infer_image(image_path):
    im = cv2.imread(image_path)
    boxes_c = detect_pnet(im, 20, 0.79, 0.6)
    print(boxes_c.shape)
    if boxes_c is None:
        return None, None

    boxes_c = detect_rnet(im, boxes_c, 0.7)
    print(boxes_c.shape)
    if boxes_c is None:
        return None, None

    boxes_c, landmark = detect_onet(im, boxes_c, 0.7)
    print(boxes_c.shape)
    if boxes_c is None:
        return None, None

    return boxes_c, landmark


if __name__ == '__main__':
    image_path = '222.jpg'
    boxes_c, landmarks = infer_image(image_path)
    img = cv2.imread(image_path)
    if boxes_c is not None:
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # 画人脸框
            cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            # 判别为人脸的置信度
            cv2.putText(img, '{:.2f}'.format(score),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 画关键点
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i]) // 2):
                cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
        cv2.imwrite("result.jpg", img)
    else:
        print('image not have face')
