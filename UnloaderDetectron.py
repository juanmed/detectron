#! /usr/bin/python3

import os
import numpy as np
import detectron2 
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import GenericMask, Visualizer
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

class UnloaderDetectron:

    def __init__(self, threshold = 0.5):
        self.cfg = get_cfg()
        self.cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # for instance segmentation
        #self.self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        #self.self.cfg.DATASETS.TRAIN = ("keypoints_train",)
        #self.self.cfg.DATASETS.TEST = ("skku_unloading_coco_test",)   # no metrics implemented for this dataset
        #self.self.cfg.DATASETS.VAL = ("skku_unloading_coco_val",)   # no metrics implemented for this dataset

        #self.self.cfg.TEST.EVAL_PERIOD = 50000
        #self.self.cfg.TEST.KEYPOINT_OKS_SIGMAS = [1.0]

        #self.self.cfg.DATALOADER.NUM_WORKERS = 6

        #self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
        self.cfg.MODEL.WEIGHTS = "./output/model_final.pth" #model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #   #
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1024   # faster, and good enough for this toy dataset
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 4 classes (box, icebox, pouch, sack)
        #   ENABLE KEYPOINT REGRESSION
        self.cfg.MODEL.KEYPOINT_ON = True
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
        self.cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 0
        #   ENABLE INSTANCE SEGMENTATION
        self.cfg.MODEL.MASK_ON =  True
        self.cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 0.5

        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = 0.02
        self.cfg.SOLVER.CHECKPOINT_PERIOD =  500
        self.cfg.SOLVER.STEPS=[]
        self.cfg.SOLVER.GAMMA = 1/128  
        # ********* Learning rate calc: https://github.com/facebookresearch/detectron2/issues/1128#issuecomment-774175041
        num_gpu = 1
        bs = (num_gpu * 2)
        self.cfg.SOLVER.BASE_LR = 0.01 * bs / 128  # pick a good LR
        # ********
        self.cfg.SOLVER.MAX_ITER = 9000   # 300 iterations seems good enough, but you can certainly train longer
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)  

        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set the testing threshold for this model
        #self.cfg.DATASETS.TEST = ("skku_unloading_coco_test", )
        self.predictor = DefaultPredictor(self.cfg)

    def forward(self, image):
        self.width, self.height = image.shape[1], image.shape[0]
        output = self.predictor(image)
        return output

    def post_process(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        #labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.height, self.width) for x in masks]
            print("Has masks")
        else:
            masks = None

        num_instances = 0
        if boxes is not None:
            boxes = Visualizer._convert_boxes(None,boxes)
            num_instances = len(boxes)
            print("Boxes:", num_instances)
        if masks is not None:
            masks = Visualizer._convert_masks(None,masks)
            if num_instances:
                assert len(masks) == num_instances
                print("Good Masks:", num_instances)
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
                print("Good Keypoints:", num_instances)
            else:
                num_instances = len(keypoints)
            keypoints = Visualizer._convert_keypoints(None,keypoints)

        dict_for_YolactSegm = {'seg_masks': [], 
                                'num_objs': num_instances, 
                                'image_size': (self.height, self.width),
                                'class_id': [], 
                                'seg_length': [], 
                                'scores': [], 
                                'bboxes': [],
                                'seg_num':[]}

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            #labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            #assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None
            scores = scores[sorted_idxs] if scores is not None else None
            classes = classes[sorted_idxs] if classes is not None else None

        for i, ( box, mask, kp, sc, cl) in enumerate(zip(boxes, masks, keypoints, scores, classes)):
            x0,y0,x1,y1 = box

            dict_for_YolactSegm['bboxes'].extend(box)
            dict_for_YolactSegm['scores'].append(sc.item())
            dict_for_YolactSegm['class_id'].append(cl.item())
            
            for j, segment in enumerate(mask.polygons):
                dict_for_YolactSegm['seg_masks'].extend(segment.tolist())
                dict_for_YolactSegm['seg_length'].append(len(segment.tolist()))
            dict_for_YolactSegm['seg_num'].append(j+1)

        return dict_for_YolactSegm        


if __name__ == "__main__":
    import cv2
    img = cv2.imread("/home/rise/Downloads/2_data_2020-11-16-15-51-17_cam1_rgb_image_rect_color.png", cv2.IMREAD_COLOR)
    detector = UnloaderDetectron()
    output = detector.forward(img)
    fields = detector.post_process(output['instances'].to("cpu"))
    
