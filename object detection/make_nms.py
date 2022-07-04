def nms(bboxes, iou_threshold, sigma=0.3):
  classes_in_img = list(set(bboxes[:,5])) # 5번째 index에 있는 값을 가지고 온다. - class! (object들 class ex.car, dog, cat..)
                        # set함수를 통해서 집합으로 만든다(순서 상관 x)
  best_bboxes = [] # return할 bound box!

  for cis in classes_in_img: # object별로 진행하기 위해서
    cls_mask = (bboxes[:,5]==cls) # 우선, object class = 0에 해당하는 것만 저장
    cls_bboxes = bboxes[cls_mask]
    while len(cls_bboxes) > 0:

      max_ind = np.argmax(cls_bboxes[:,4]) # score column -> 제일 큰 값
      best_bbox = cls_bboxes[max_ind]
      best_bboxes.append(best_bbox) # 추가
      cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]]) 
      # best box를 빼고, 나머지 내용은 합친다.

      iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:,:4]) 
      # 전에 정의한 bboxes_iou함수를 통해서 max를 제외한 cls_bboxes의 bound box 갯수만큼 벡터로 나온다.
      weight = np.ones((len(iou), ), dtype=np.float32) # iou의 갯수만큼 1값을 채워서 만든다.

      iou_mask = iou > iou_threshold # iou_threshold =0.4를 주어졌다면, 그 값보다 큰 값들을 true
      weight[iou_mask] =0.0 # 큰 값들을 제거해주기 위해서 이렇게 저장해둔다. -> why? 겹치기 때문에 

      cls_bboxes[:,4] = cls_bboxes[:,4] * weight
      score_mask = cls_bboxes[:,4] > 0.
      cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes
