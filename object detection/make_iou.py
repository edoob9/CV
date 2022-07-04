# ver.1
import numpy as np 

def compute_iou(cand_box, gt_box):

    # Calculate intersection areas
    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.minimum(cand_box[2], gt_box[2])
    y2 = np.minimum(cand_box[3], gt_box[3])
    
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area - intersection
    
    iou = intersection / union
    return iou
  
# ver.2
def bbox_iou(boxes1, boxes2):
  boxes1_area = boxes1[..., 2] * boxes1[..., 3] # width, heigh값 -> 넓이 계산
  boxes2_area = boxes2[..., 2] * boxes2[..., 3] # width, heigh값

  boxes1 = tf.concat([boxes1[...,:2]- boxes1[...,2:] *0.5,
                      [boxes1[...,:2]- boxes1[...,2:] *0.5], axis=-1)
  boxes2 = tf.concat([boxes2[...,:2]- boxes2[...,2:] *0.5,
                      [boxes2[...,:2]- boxes2[...,2:] *0.5], axis=-1)
  left_up = tf.maximum(boxes1[...,:2], boxes2[...,:2])
  rigth_down = tf.minimum(boxes1[...,2:], boxes2[...,2:])

  inter_section = tf.maximum(right_down - left_up, 0.0) # 겹치는 부분의 내부 영역의 좌표 구하기(포함하고 있는 거 = 교집합 느낌이라고 설명할 수 있다.)
  inter_area = inter_selection[..., 0] * inter_selection[..., 1] 
  union_area = boxes1_area + boxes2_area - inter_area

  return 1.0 * inter_area / union_area
