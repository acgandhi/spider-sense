import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, bbox_iou
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from icecream import ic


def spider_sense(headDet, weapDet, im0, thres):
    detections = [False, False, 0, False]
    # print(len(headDet), headDet)
    # print(len(weapDet), weapDet)
    headThres = {2: 0.21, 3: 0.15}
    # removing detections that don't meet the necessary width threshold
    for i in range(0, len(headDet[-1])):
        width = float((headDet[-1][i][2] - headDet[-1][i][0]) / im0.shape[1])
        if width < headThres[thres]:
            headDet[-1][i] = False
    # Checking for head width and weapons and doing context check if valid
    if len(headDet[-1]) and len(headDet) == 5:
        detections[2] += len(headDet[-1])
        for detection in headDet[-1]:
            if type(detection) == bool:
                continue
            context = 0
            tempDet = detection
            for i in range(3, -1, -1):  # each frame
                for j in range(0, len(headDet[i])):  # each detection in frame
                    if bbox_iou(tempDet, headDet[i][j], DIoU=True) >= 0.1542:
                        context += 1
                        tempDet = headDet[i][j]
                        break
                if context != (4 - i):
                    break
            if context >= 3:
                detections[0] = True
                break
    elif len(headDet) < 5:
        for detection in headDet[-1]:
            width = float((detection[2] - detection[0]) / im0.shape[1])
            if width >= headThres[thres]:
                detections[3] = True
                break
    noContext = 0
    if len(weapDet[-1]) > 0 and len(weapDet) == 5:
        for detection in weapDet[-1]:
            context = 0
            tempDet = detection
            for i in range(3, -1, -1):                  # each frame
                for j in range(0, len(weapDet[i])):     # each detection in frame
                    if tempDet[5] == weapDet[i][j][5] and bbox_iou(tempDet, weapDet[i][j], DIoU=True) >= 0.1542:
                        context += 1
                        tempDet = weapDet[i][j]
                        break
                if context < (3-i):
                    break
            if context >= 3:
                detections[1] = True
                break

    return detections


def detect(save_img=False):   
    numDet = []
    source, weights, weights2, view_img, save_txt, imgsz, thres = opt.source, opt.weights, opt.weights2, opt.view_img, opt.save_txt, opt.img_size, opt.headThres
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(opt.project)
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load models
    model = attempt_load(weights, map_location=device)
    model2 = attempt_load(weights2, map_location=device)
    stride = int(model.stride.max())  # model strides
    stride2 = int(model2.stride.max()) # model 2 strides
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
        model2.half() # to FP16 too

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride1)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    names2 = model2.module.names if hasattr(model2, 'module') else model2.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    colors2 = [[random.randint(0, 255) for _ in range(3)] for _ in names2]

    # Run inference
    t0 = time.time()
    numWeapons = 0
    headDet = []
    weapDet = []
    for path, img, im0s, vid_cap in dataset:        
        print("\n")
        t1 = time_synchronized()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Do first round of predictions
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            numDet.append(len(det))
            weapDet.append(det.clone())
            if len(weapDet) > 5:
                weapDet.pop(0)
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i], dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        
            if len(det):
                # Rescale boxes from img_size to im0 size
                print(img.shape[2:], det[:, :4], im0.shape)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # for detection in det[:, :4]:
                # print(str((detection[2]-detection[0])/im0.shape[1]))

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names2[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names2[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        
        print("2nd Round")
        model = model2
        names = names2
        colors = colors2

        # Do second round of predictions
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model2.parameters())))  # run once

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            numWeapons += len(det)
            headDet.append(det.clone())
            if len(headDet) > 5:
                headDet.pop(0)
            numDet.append(len(det))
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i], dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for detection in det[:, :4]:
                    print(str((detection[2] - detection[0]) / im0.shape[1]))

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Checking for Spider-Sense
            sense = spider_sense(headDet, weapDet, im0, thres)
            print(len(headDet))
            print(sense[3])
            print(len(weapDet[0]) > 0)
            if sense[0] or sense[1]:
                cv2.putText(im0, "Spider-Sense Tingling!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
                print("BIG PPPPPPPPPPPPPPPOOOOOOOOOOOOOPPPPPPP")

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')
    # print(numWeapons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--weights2', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--headThres', type=int, default=2)
    opt = parser.parse_args()
    print(opt)
    # check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()