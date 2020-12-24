import argparse
import logging
import time
from PIL import Image

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    cap = cv2.VideoCapture(args.video)
    #w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    name = args.video.split("/")[-1].split(".")[0]
    framerate = int(cap.get(cv2.CAP_PROP_FPS))
    print("FRAMERATE:", framerate)
    out = cv2.VideoWriter('./results/'+name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), framerate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    print("GOOOOOOD SOOOOOOO FAAAAAAAARRRRRRRRRRR \n")
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while True:
        ret_val, image = cap.read()
        if not ret_val:
            break

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #out.write(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        out.write(image)
        print("running")       

        #cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
    print("CREATE VIDEOOOOOO \n")
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    print("Showing " + name + " HELLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")

logger.debug('finished+')
