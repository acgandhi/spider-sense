import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, bbox_iou
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from icecream import ic


def dist_check(tempDet, otherDet):
    perim = 2 * (tempDet[3] - tempDet[1]) + 2 * (tempDet[2] - tempDet[0])
    width = otherDet[0]+(otherDet[2]-otherDet[0])/2 - tempDet[0]+(tempDet[2]-tempDet[0])/2
    height = otherDet[0]+(otherDet[2]-otherDet[0])/2 - tempDet[0]+(tempDet[2]-tempDet[0])/2
    return math.sqrt(width**2 + height**2) < perim/2


def genDet(oldFrame, frame, secDet, mask):
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # list of inner points
    innerPoints = []
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # generating points
    offset = 10
    for i in range(0, len(secDet)):
        inner = np.zeros((4, 1, 2), dtype=float)
        width = secDet[i][2] - secDet[i][0]
        height = secDet[i][3] - secDet[i][1]
        centerX = secDet[i][0] + width/2
        centerY = secDet[i][1] + height/2
        inner[0][0][0] = centerX - (width / offset)
        inner[0][0][1] = centerY - (height / offset)
        inner[1][0][0] = centerX + (width / offset)
        inner[1][0][1] = centerY - (height / offset)
        inner[2][0][0] = centerX - (width / offset)
        inner[2][0][1] = centerY + (height / offset)
        inner[3][0][0] = centerX + (width / offset)
        inner[3][0][1] = centerY + (height / offset)
        innerPoints.append(inner)
    newBoxes = []
    for point, det in zip(innerPoints, secDet):
        old_gray = cv2.cvtColor(cv2.cvtColor(oldFrame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, point.astype('float32'), None, **lk_params)
        if opt.flowShow:
            for i,(new,old) in enumerate(zip(p1, point)):
                a,b = new.ravel()
                c,d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        newCenter = [sum([pt[0][0] for pt in p1]) / 4, sum([pt[0][1] for pt in p1]) / 4]
        width = det[2] - det[0]
        height = det[3] - det[1]
        newBoxes.append(torch.Tensor([newCenter[0] - width / 2, newCenter[1] - height / 2, newCenter[0] + width / 2, newCenter[1] + height / 2, det[4], det[5]]))
    if type(secDet) != list:
        secDet = secDet.tolist()
    secDet.extend(newBoxes)
    

def spider_sense(headDet, weapDet, frames, im0, thres, mask):
    detections = [False, False]
    headThres = {2: 0.21, 3: 0.15, 4: 0.09}        
    
    # removing head detections that don't meet the necessary width threshold
    headDet[-1] = [det for det in headDet[-1] if float(2 * (det[2] - det[0]) / im0.shape[1]) >= headThres[thres]]

    # removing weapon detections that are too wide
    weapDet[-1] = [det for det in weapDet[-1] if float(2 * (det[2] - det[0]) / im0.shape[1]) < 0.5]

    # adding new detections from second last with current if >= 2 detections
    if len(headDet) >= 2 and len(frames) >= 2:
        print("Generating New Detections", len(headDet[-2]), len(weapDet[-2]))
        genDet(frames[-2], frames[-1], headDet[-2], mask)
        genDet(frames[-2], frames[-1], weapDet[-2], mask)
        print("Done:", len(headDet[-2]), len(weapDet[-2]))

    # Doing context check on remaining head and weapons and changing detections if needed
    if len(headDet[-1]) > 0 and len(headDet) == opt.filterLen:
        for detection in headDet[-1]:
            context = 0
            tempDet = detection
            for i in range(opt.filterLen-2, -1, -1):  # each frame
                for j in range(0, len(headDet[i])):  # each detection in frame
                    if bbox_iou(tempDet, headDet[i][j], DIoU=True) >= 0.1542:
                    #if dist_check(tempDet, headDet[i][j]):
                        context += 1
                        tempDet = headDet[i][j]
                        break
                if context != (opt.filterLen - i):
                    break
            if context >= opt.filterLen-2:
                detections[0] = True
                break
    noContext = 0
    if len(weapDet[-1]) > 0 and len(weapDet) == opt.filterLen:
        for detection in weapDet[-1]:
            context = 0
            tempDet = detection
            for i in range(opt.filterLen-2, -1, -1):  # each frame
                for j in range(0, len(weapDet[i])):  # each detection in frame
                    if tempDet[5] == weapDet[i][j][5] and bbox_iou(tempDet, weapDet[i][j], DIoU=True) >= 0.1542:
                    #if tempDet[5] == weapDet[i][j][5] and dist_check(tempDet, weapDet[i][j]):
                        context += 1
                        tempDet = weapDet[i][j]
                        break
                if context < (opt.filterLen - i):
                    break
            if context >= opt.filterLen-2:
                detections[1] = True
                break

    return detections


@torch.no_grad()
def detect(save_img=False):
    source, weights, weights2, view_img, save_txt, imgsz, thres = opt.source, opt.weights, opt.weights2, opt.view_img, opt.save_txt, opt.img_size, opt.headThres
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(opt.project)
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load models
    model1 = attempt_load(weights, map_location=device)
    model2 = attempt_load(weights2, map_location=device)
    stride1 = int(model1.stride.max())  # model strides
    stride2 = int(model2.stride.max())  # model 2 strides
    names1 = model1.module.names if hasattr(model1, 'module') else model1.names
    names2 = model2.module.names if hasattr(model2, 'module') else model2.names
    imgsz = check_img_size(imgsz, s=stride1)  # check img_size
    if half:
        model1.half()  # to FP16
        model2.half()  # to FP16 too

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        if opt.saveWebcam:
            save_img = True
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride1)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride1)

    # Run inference
    numFrames = 1
    t0 = time.time()
    numWeapons = 0
    headDet = []
    weapDet = []
    frames = []
    mask = None
    for path, img, im0s, vid_cap in dataset:
        print("\nFrame:", numFrames)
        if webcam:
            print("FPS", dataset.fps)
        t1 = time_synchronized()

        # Adding to frame
        if(len(img.shape) >= 4):
            myImg = np.dstack((img[0, 0], img[0, 1], img[0, 2]))
        else:
            myImg = np.dstack((img[0], img[1], img[2]))
        
        # Creating mask
        if mask is None:
            mask = np.zeros_like(myImg)
        
        # Appending frames
        frames.append(myImg)
        if len(frames) > opt.filterLen:
            frames.pop(0)

        # Starting with the actual detections
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Do first round of predictions
        model = model1  # set pointer to model1
        names = names1

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            weapDet.append(det.clone())
            if len(weapDet) > opt.filterLen:
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
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # for detection in det[:, :4]:

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
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=Path(save_path + '_crops') / names[c] / f'{p.stem}.jpg', BGR=True)


        print("2nd Round")
        model = model2
        names = names2

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            numWeapons += len(det)
            headDet.append(det.clone())
            if len(headDet) > opt.filterLen:
                headDet.pop(0)
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i], dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

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
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        width = round(float((xyxy[2] - xyxy[0])/im0.shape[1]), 2)
                        plot_one_box(xyxy, im0, label=label + " " + str(width), color=colors(c, True), line_thickness=opt.line_thickness)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Checking for Spider-Sense
            sense = spider_sense(headDet, weapDet, frames, im0, thres, mask)
            if sense[0] or sense[1]:
                cv2.putText(im0, "Spider-Sense Tingling!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                elif webcam:
                    if vid_path != save_path + ".mp4":
                        vid_path = save_path + ".mp4"
                        print("Save Path: ", save_path)
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                            
                        fourcc = 'mp4v'
                        fps = dataset.fps
                        w = dataset.w
                        h = dataset.h
                        vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if opt.flowShow:
                        thisMask = cv2.resize(mask, (im0.shape[1], im0.shape[0]))
                        im0 = cv2.add(im0, thisMask)
                    vid_writer.write(im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        print("Save Path: ", save_path)
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    if opt.flowShow:
                        thisMask = cv2.resize(mask, (im0.shape[1], im0.shape[0]))
                        im0 = cv2.add(im0, thisMask)
                    vid_writer.write(im0)
          
        # Checking break condition
        if numFrames == opt.maxFrames:
            break        
        numFrames += 1


    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()  # release previous video writer
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
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--save-crop', default=False)
    parser.add_argument('--headThres', type=int, default=4)
    parser.add_argument('--filterLen', type=int, default=5)
    parser.add_argument('--saveWebcam', type=bool, default=False)
    parser.add_argument('--flowShow', type=bool, default=False)
    parser.add_argument('--maxFrames', type=int, default=-1)
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