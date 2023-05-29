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

    # Set Dataloader

    view_img = check_imshow()
    print("view img is ", view_img)
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

    t0 = time.time()

   # DATALOADER OUTPUT
    for path, img, im0s, vid_cap in dataset:
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
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, 0.25, 0.45, classes=opt.classes)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
            ), dataset.count

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label,
                                 color=colors[int(cls)], line_thickness=1)

                    # print(xyxy[0].item()) # xmin
                    print("x1 = %.1f , y1 = %.1f , x2 = %.1f , y2 = %.1f" % (xyxy[0].item(
                    ), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()))

            # Tampilkan ke layar
            cv2.imshow(str(p), im0)

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
