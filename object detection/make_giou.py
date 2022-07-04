# < Generalized IOU >
# - 두 box간 교집합이 없으면, IOU는 0이 되므로 얼마나 오차가 있었는지 알기 어려운 점을 보완한 지표 
# => 어느 정도까지 틀렸는지 확인할 수 있는거를 커버할 수 있게 IOU로 결과를 받으면 알 수 없는 수치를 알기 위해서 
# 1) A, B box는 거리가 비교적 좀 많이 멀리 존재한다. -> error값이 높다.
# 2) C, D box는 거리가 1)과 비교적 가깝게 설정되어있다. 그렇다면, 1번보다 error가 낮다. -> 조금만 업데이트하면 된다.

def bbox_giou(boxes1, boxes2):
  boxes1 = tf.concat([boxes1[..., :2] * boxes1[..., 2:] *0.5,
                     [boxes1[..., :2] - boxes1[...,2:] *0.5], axis=-1)
  boxes2 = tf.concat([boxes2[...,:2]- boxes2[...,2:] *0.5,
                      [boxes2[...,:2]- boxes2[...,2:] *0.5], axis=-1)
  
  boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., :2]),
                      tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
  boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., :2]),
                      tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

  boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] * boxes1[..., 1])
  boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] * boxes2[..., 1])

  left_up = tf.maximum(boxes1[...,:2], boxes2[...,:2])
  rigth_down = tf.minimum(boxes1[...,2:], boxes2[...,2:])

  inter_section = tf.maximum(right_down - left_up, 0.0) # 겹치는 부분의 내부 영역의 좌표 구하기(포함하고 있는 거 = 교집합 느낌이라고 설명할 수 있다.)
  inter_area = inter_selection[..., 0] * inter_selection[..., 1] 
  union_area = boxes1_area + boxes2_area - inter_area # 합집합 크기

  enclose_left_up = tf.minimum(boxes1[...,:2], boxes2[...,:2])
  enclose_right_down = tf.maximum(boxes1[...,2:], boxes2[...,2:])
  enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
  enclose_area = enclose[..., 0] * enclose[..., 1] # GT와 prediction을 포괄하는 가장 작은 박스
  giou = iou - 1.0* (enclose_area - union_area) / enclose_area
 
  return giou
