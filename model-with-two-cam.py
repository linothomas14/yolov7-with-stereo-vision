import argparse
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from random import randint
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer, set_logging

from utils.plots import plot_one_box
from utils.torch_utils import select_device

import triangulation as tri
import calibration

from tkinter import *
from PIL import Image, ImageTk


def detect():
    source, weights,  imgsz = arg.source, arg.weights, arg.img_size

    root = Tk()
    root.title("SISTEM KLASIFIKASI KEMATANGAN BUAH PEPAYA DAN ESTIMASI JARAK")

    title_label = Label(
        root, text="SISTEM KLASIFIKASI KEMATANGAN BUAH PEPAYA DAN ESTIMASI JARAK", wraplength=1000, font=("Helvetica", 20))
    title_label.grid(row=0, column=0, columnspan=2)

    label1 = Label(root, pady=10, padx=10)
    label1.grid(row=1, column=0)

    label2 = Label(root, pady=10, padx=10)
    label2.grid(row=1, column=1)

    # Stereo vision setup parameters
    B = 8  # Distance between the cameras [cm]
    alpha = 75  # Camera field of view in the horizontal plane [degrees]

    center_point_left = 0, 0  # titik tengah dari objek yang dideteksi
    center_point_right = 0, 0
    result_left = []  # menampung hasil deteksi dari kamera kiri
    result_right = []  # menampung hasil deteksi dari kamera kanan

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[252, 123, 3], [3, 64, 1]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

   # DATALOADER OUTPUT
    for path, img, im0s, vid_cap in dataset:

        # Implement Kalibrasi
        # im0s[1], im0s[0] = calibration.undistortRectify(im0s[1], im0s[0])
        # img[1], img[0] = calibration.undistortRectify(img[1], img[0])

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, 0.30, 0.45)

        # Process detections
        i_left = 0
        det_left = pred[i_left]  # left cam

        im0_left = im0s[i_left].copy()

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
                """
                xywh_left dalam skala 0-1
                [0] = koordinat tengah x objek
                [1] = koordinat tengah y objek
                [2] = lebar objek
                [3] = tinggi objek
                """

                boundBox_left = int(xywh_left[0] * w), int(xywh_left[1] * h), int(
                    xywh_left[2] * w), int(xywh_left[3] * h)
                print(xyxy[0], xyxy[1],
                      xyxy[2], xyxy[3])
                # tengah X dan tengah Y
                center_point_left = (boundBox_left[0], boundBox_left[1])

                result_left.append(
                    {"class": names[int(cls)], "conf": f'{conf:.2f}', "center_point": center_point_left, "coor": xyxy, "color": colors[int(cls)]})

            # mengurutkan item dari objek paling kiri
            result_left = sorted(
                result_left, key=lambda x: x["center_point"][0])

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

                center_point_right = (boundBox_right[0], boundBox_right[1])

                label = f'{names[int(cls)]} {conf:.2f}'

                result_right.append(
                    {"class": names[int(cls)], "conf": f'{conf:.2f}', "center_point": center_point_right, "coor": xyxy, "color": colors[int(cls)]})

            # mengurutkan item dari objek paling kiri
            result_right = sorted(
                result_right, key=lambda x: x["center_point"][0])

        if len(det_left) <= 0 | len(det_right) <= 0:
            print("TRACKING LOST")

        elif len(result_left) != len(result_right):

            print("Jumlah objek tidak sama, kiri ",
                  result_left, "kanan ", result_right)

        else:
            for item_left, item_right in zip(result_left, result_right):
                depth = tri.find_depth(
                    (item_right["center_point"]), (item_left["center_point"]), im0_right, im0_left, B, alpha)

                item_right["depth"] = depth

                label = f'{item_right["class"]} {item_right["conf"]}, {item_right["depth"]} cm'

                plot_one_box(item_right["coor"], im0_right, label=label,
                             color=item_right["color"], line_thickness=1)

        # TAMPILKAN KE LAYAR
        frames = [im0_left, im0_right]
        labels = [label1, label2]
        for frame, label in zip(frames, labels):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.image = photo

        root.update()

        result_right.clear()  # reset isi dari array
        result_left.clear()  # reset isi dari array

        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    # parser.add_argument('--device', default='',
    #                     help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    arg = parser.parse_args()
    print(arg)

    with torch.no_grad():
        detect()
        strip_optimizer(arg.weights)
