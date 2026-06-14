import numpy as np

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def nms(detections, iou_threshold=0.45):
    """
    Non-Maximum Suppression (NMS) for overlapping boxes.
    Expects list of dicts with 'bbox_xyxy' and 'confidence'.
    """
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if compute_iou(best['bbox_xyxy'], d['bbox_xyxy']) < iou_threshold
        ]
        
    return keep
