import argparse
import time
import os
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from random import randint
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.download_weights import download

import triangulation as tri
import calibration


def detect():
    source, weights, save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Stereo vision setup parameters
    frame_rate = 120  # Camera frame rate (maximum at 120 fps)
    B = 8  # Distance between the cameras [cm]
    f = 3.6  # Camera lens's focal length [mm]
    alpha = 75  # Camera field of view in the horizontal plane [degrees]

    center_point_left = 0, 0  # titik tengah dari objek yang dideteksi
    center_point_right = 0, 0

    result_left = []  # menampung hasil deteksi dari kamera kiri
    result_right = []  # menampung hasil deteksi dari kamera kanan
    # Set Dataloader

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

   # DATALOADER OUTPUT
    for path, img, im0s, vid_cap in dataset:
        # img[0] = cv2.cvtColor(img[0], cv2.COLOR_BGR2BGRA)
        # img[1] = cv2.cvtColor(img[1], cv2.COLOR_BGR2BGRA)

        im0s[1], im0s[0] = calibration.undistortRectify(im0s[1], im0s[0])
        # print(type(img[0]))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference

        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, 0.30, 0.45, classes=opt.classes)

        # Process detections
        det_left = pred[0]  # left cam
        i_left = 0

        p_left, s_left, im0_left = path[i_left], '%g: ' % i_left, im0s[i_left].copy(
        )
        gn_left = torch.tensor(im0_left.shape)[[1, 0, 1, 0]]

        h, w, c = im0_left.shape

        if len(det_left):
            # Rescale boxes from img_size to im0 size
            det_left[:, :4] = scale_coords(
                img.shape[2:], det_left[:, :4], im0_left.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det_left):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0_left, label=label,
                             color=colors[int(cls)], line_thickness=1)

                xywh_left = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                       ) / gn_left).view(-1).tolist()

                boundBox_left = int(xywh_left[0] * w), int(xywh_left[1] * h), int(
                    xywh_left[2] * w), int(xywh_left[3] * h)

                center_point_left = (boundBox_left[0], boundBox_left[1])

                result_left.append(
                    {"class": names[int(cls)], "conf": f'{conf:.2f}', "center_point": center_point_left, "coor": xyxy})

            # mengurutkan item dari objek paling kiri
            result_left = sorted(
                result_left, key=lambda x: x["center_point"][0])

            # print("cam : ", p_left, "x1 = %.1f , y1 = %.1f , x2 = %.1f , y2 = %.1f" % (xyxy[0].item(
            # ), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))

        det_right = pred[1]  # right cam
        i_right = 1

        p_right, s_right, im0_right = path[i_right], '%g: ' % i_right, im0s[i_right].copy(
        )

        gn_right = torch.tensor(im0_right.shape)[[1, 0, 1, 0]]

        if len(det_right):
            # Rescale boxes from img_size to im0 size
            det_right[:, :4] = scale_coords(
                img.shape[2:], det_right[:, :4], im0_right.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det_right):

                xywh_right = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                        ) / gn_right).view(-1).tolist()

                boundBox_right = int(xywh_right[0] * w), int(xywh_right[1] * h), int(
                    xywh_right[2] * w), int(xywh_right[3] * h)

                # center_point_right = (
                #     boundBox_right[0] + int(boundBox_right[2] / 2), boundBox_right[1] + int(boundBox_right[3] / 2))

                center_point_right = (boundBox_right[0], boundBox_right[1])

                label = f'{names[int(cls)]} {conf:.2f}'

                result_right.append(
                    {"class": names[int(cls)], "conf": f'{conf:.2f}', "center_point": center_point_right, "coor": xyxy})

                # plot_one_box(xyxy, im0_right, label=label,
                #              color=colors[int(cls)], line_thickness=1)

            # mengurutkan item dari objek paling kiri
            result_right = sorted(
                result_right, key=lambda x: x["center_point"][0])

            # print("Jarak : ",  str(depth) + " cm")


        if len(det_left) <= 0 | len(det_right) <= 0:
            print("TRACKING LOST")

        elif len(result_left) != len(result_right):

            print("Jumlah objek tidak sama, kiri ",
                  result_left, "kanan ", result_right)

        else:
            for item_left, item_right in zip(result_left, result_right):
                depth = tri.find_depth(
                    (item_right["center_point"]), (item_left["center_point"]), im0_right, im0_left, B, f, alpha)

                item_right["depth"] = depth

                label = f'{item_right["class"]} {item_right["conf"]}, {item_right["depth"]} cm'

                plot_one_box(item_right["coor"], im0_right, label=label,
                             color=colors[int(cls)], line_thickness=1)

        cv2.imshow(str(p_left), im0_left)
        cv2.imshow(str(p_right), im0_right)

        result_right.clear()  # reset isi dari array
        result_left.clear()  # reset isi dari array

        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='best.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true',
                        help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='object_tracking',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)

    #check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(str(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        detect()
        strip_optimizer(opt.weights)
