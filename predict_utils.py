  
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from random import randint

import cv2
import mmcv
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmcv.utils import print_log
from mmcv.image import imread, imwrite

from mmdet.utils import get_root_logger


class PostProcessor():
    def __init__(self,
                classes,
                score_thr=0.3):
        self.classes = classes
        self.num_classes = len(classes)
        if self.num_classes == 10:
            self.box_real_class = [0, 1, 1, 2, 3, 4, 5, 6, 5, 6]
        elif self.num_classes == 12:
            self.box_real_class = [0, 1, 1, 2, 3, 4, 5, 6, 2, 5, 0, 6]
        self.score_thr = score_thr
        self.thickness = 1
        self.font_scale = 0.5
        self.win_name = ''
        self.wait_time = 0
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255),
                        (255, 0, 255), (255, 255, 255), (0, 0, 0), (255, 0, 128), (0, 191, 255),
                        (10, 255, 128), (191, 255, 0), (255, 191, 0), (255, 128, 10), (50, 152, 89)]
        self.makeColors()
        self.iitpID = 1
        self.iitpJson = {'annotations':[]}

    def makeColors(self):
        if len(self.colors) >= self.num_classes:
            return
        else:
            while len(self.colors) < self.num_classes:
                self.colors.append((randint(20, 230), randint(20, 230), randint(20, 230)))
            return

    def saveResult(self,
                    img,
                    result,
                    show=False,
                    out_file=None):
        img, bboxes, labels = self.extractInfo(img, result, show=False, out_file=out_file)
        
        # draw bounding boxes
        return self.imshow_det_bboxes(img, bboxes, labels, show=show, out_file=out_file)

    def labelChanger(self, labels):
        appliedLabels = []
        if self.num_classes == 10:
            for i in labels:
                if i == 8:
                    if 3 in labels or 4 in labels:
                        i = 3
                if i == 9:
                    if 3 in labels:
                        i = 3 
                    elif 4 in labels:
                        i = 0 
                i = self.box_real_class[i]
                i += 1
                appliedLabels.append(i)
        elif self.num_classes == 12:
            appliedLabels = [self.box_real_class[i]+1 for i in labels]
        else:
            print('Unexpected # class')
            raise ValueError

        return appliedLabels

    def saveIitp(self, img, imgPath, result):
        _, bboxes, labels = self.extractInfo(img, result, show=False, out_file=None)
        bboxes, labels = self.iitpProcess(bboxes, labels)
        if len(labels) < 1:
            return False
        return self.annoMaker(imgPath, bboxes, labels)

    def annoMaker(self, imgPath, bboxes, labels, labelChanger=True):
        anno = {}
        anno['id'] = self.iitpID
        self.iitpID += 1
        if labelChanger:
            labels = self.labelChanger(labels)
        fileName = imgPath.split('/')[-1]
        anno['file_name'] = fileName
        anno['object'] = []
        for box, label in zip(bboxes, labels):
            anno['object'].append({
                'box': box,
                'label': 'c'+str(label)
                })
        self.iitpJson['annotations'].append(anno)

        return labels

    def iitpProcess(self, bboxes, labels):
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        processedBoxes = []
        for box in bboxes:
            box = box.tolist()
            box.pop()
            box = list(map(int, box))
            processedBoxes.append(box)

        return processedBoxes, labels

    def bb_intersection_over_union(self, bboxes, labels, box_scores):
        # determine the (x, y)-coordinates of the intersection rectangle
        best_indexes = []

        for i in range(0, len(bboxes) - 1):

            best_iou = -1
            best_list = []

            for j in range(i + 1 , len(bboxes)):
                xA = max(bboxes[i][0], bboxes[j][0])
                yA = max(bboxes[i][1], bboxes[j][1])
                xB = min(bboxes[i][2], bboxes[j][2])
                yB = min(bboxes[i][3], bboxes[j][3])
                # compute the area of intersection rectangle
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                # compute the area of both the prediction and ground-truth
                # rectangles
                boxAArea = (bboxes[i][2] - bboxes[i][0] + 1) * (bboxes[i][3] - bboxes[i][1] + 1)
                boxBArea = (bboxes[j][2] - bboxes[j][0] + 1) * (bboxes[j][3] - bboxes[j][1] + 1)
                # compute the intersection over union by taking the intersection
                # area and dividing it by the sum of prediction + ground-truth
                # areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
                if iou > best_iou:
                    best_iou = iou
                    best_list = [i , j, best_iou]

                    best_indexes.append(best_list)

        index = []
        for best_index in best_indexes:
            if best_index[2] > 0.98: # best_iou
                if box_scores[best_index[0]] > box_scores[best_index[1]]:
                    index.append(best_index[1])

                else :
                    index.append(best_index[0])

        index = set(index)
        index = sorted(list(index), reverse=True)

        for i in index :
            if box_scores[i] < self.score_thr + 0.05:
                bboxes = np.delete(bboxes, i, axis = 0)
                labels = np.delete(labels, i, axis = 0)
                box_scores = np.delete(box_scores, i, axis = 0)

        return bboxes, labels, box_scores

    def cropBoxes(self, img, result, out_file=None):
        img, bboxes, labels = self.extractInfo(img, result, show=False, out_file=out_file)
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)
        box_scores = []

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            box_scores = scores[inds]

        img = np.ascontiguousarray(img)
        croppedImgs = []
        out_label = []

        if len(labels) > 1:
            bboxes, labels, box_scores = self.bb_intersection_over_union(bboxes, labels, box_scores)

        # path to save cropped image if save
        # splitPath = out_file.split('/')
        # fileName = splitPath.pop(-1).split('.')[0]

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):

            # !!!!!!!!!!! ~ Except class cap(8) or label(9) ~ !!!!!!!!!!!!
            if label != 8 and label != 9:
                
                bbox_int = bbox.astype(np.int32)
                heightRange = (bbox_int[1], bbox_int[3])
                widthRange = (bbox_int[0], bbox_int[2])

                dst = img.copy()

                center_x = int(int(bbox_int[0]) - int(bbox_int[0])*0.15)
                center_y = int(int(bbox_int[1]) - int(bbox_int[0])*0.15)
                width = int(int(bbox_int[2]) + int(bbox_int[2])*0.15)
                height = int(int(bbox_int[3]) + int(bbox_int[3])*0.15)

                dst = dst[center_y:height, center_x:width]

                # dst = dst[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]

                croppedImgs.append(dst)

            out_label.append(label)

            # save cropped image            
            # out_file = splitPath.copy()
            # out_file.append(fileName+'_'+str(idx)+'.jpg')
            # out_file = '/'.join(out_file)
            # if out_file is not None:
            #     imwrite(dst, out_file)
        
        out_label = self.labelChanger(out_label)

        return croppedImgs, out_label

    def extractInfo(self,
                    img,
                    result,
                    show=False,
                    out_file=None):

        # batch_size = len(result)
        # print('batch_size : ', batch_size)
        # print('result : ', len(result[0]), result)
        
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
                # print('check msrcnn : ', len(segm_result))
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            # print('check segm_result is not None')
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > self.score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        # if not (show or out_file):
        #     return img

        return img, bboxes, labels

    def imshow_det_bboxes(self,
                        img,
                        bboxes,
                        labels,
                        show=True,
                        out_file=None):

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        img = np.ascontiguousarray(img)
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, self.colors[label], thickness=self.thickness)
            label_text = self.classes[
                label] if self.classes is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - (label*2*randint(0, 1))),
                        cv2.FONT_HERSHEY_COMPLEX, self.font_scale, self.colors[label])

        if show:
            imshow(img, self.win_name, self.wait_time)
        if out_file is not None:
            imwrite(img, out_file)
        return img





    def get_key_object(self, img, result, out_file=None):
        img, bboxes, labels = self.extractInfo(img, result, show=False, out_file=out_file)

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        box_scores = []

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            box_scores = scores[inds]

        if len(labels) > 1:
            bboxes, labels, box_scores = self.bb_intersection_over_union(bboxes, labels, box_scores)
        p = self.key_object(bboxes, labels)

        return labels, p

    def key_object(self, bboxes, labels):
        # set debug mode
        debug = False

        bbox = []
        if len(labels) > 1:
            for idx_box, ibox in enumerate(bboxes):
                bbox.append([ibox[0],ibox[1],ibox[2],ibox[3],ibox[4],labels[idx_box]])
        elif len(labels) == 1:
            bbox.append([bboxes[0][0],bboxes[0][1],bboxes[0][2],bboxes[0][3],bboxes[0][4],labels[0]])

        bounding_box = sorted(bbox, key=lambda k: k[0])
        if debug == True:
            print('sort: ', bounding_box)
        q = []
        p = []
        flag = 0
        for bidx, bi in enumerate(bounding_box):
            if bidx == 0 or len(q)==0:
                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
            else:
                ck = 0
                cj = 1
                flag2 = 0
                for jdx in range(0,len(q)):
                    if debug == True:
                        print('ck: ',ck)
                        print('jdx: ',jdx)
                    qsz = len(q)
                    if qsz == 1:
                        bk = q[0]
                    elif ck >= 1:
                        bk = q[ck-cj]
                    else:
                        #bk = q[jdx]
                        bk = q[jdx]

                    if debug == True:
                        print('bi: ', bi)
                        print('size of LL: ', bk.size())
                        print('bk (Q) :', bk.selectNode(0))
                        print('size of q',len(q))
                        for iddd in q:
                            print('now q: ', iddd.selectNode(0))


                    iou = self.get_iou(bk.selectNode(0).data, bi)
                    bk_area = (bk.selectNode(0).data[2]-bk.selectNode(0).data[0])*(bk.selectNode(0).data[3]-bk.selectNode(0).data[1])
                    bi_area = (bi[2]-bi[0])*(bi[3]-bi[1])
                    if debug == True:
                        print('iou',iou)
                        print('bk_area',bk_area)
                        print('bi_area',bi_area)

                    #print('iou/((bi_area/bk_area)+1e-6)',iou/((bi_area/bk_area)+1e-6))

                    #print('(bk.selectNode(0).data[1]/(bi[1]+1e-6))',(bk.selectNode(0).data[1]/(bi[1]+1e-6)))
                    #print('bk.selectNode(0).data[3]/(bi[3]+1e-6)',bk.selectNode(0).data[3]/(bi[3]+1e-6))

                    if iou >= 0.99 and iou/((bi_area/bk_area)+1e-6) >= 0.99:
                        q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                        break


                    #print('bi[1]/(bk.selectNode(0).data[3]+1e-6)',bi[1]/(bk.selectNode(0).data[3]+1e-6))

                    #if bi_area > bk_area:
                        #print('area: ', bk_area/bi_area)
                    # case 1
                    # bi_xmin >> bk_xmax
                    if bi[0] > bk.selectNode(0).data[2]:
                        if debug == True:
                            print('case 1')
                        # delete bk from Q
                        p.append(q.pop(0))
                        #if ck == 0:
                        if jdx == (len(q)) or len(q) == 0:
                            if int(bk.selectNode(0).data[5]) == 8 and bk_area < bi_area and 0.98<bk.selectNode(0).data[3]/(bi[1]+1e-6):
                                bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                break
                            else:
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                break
                        else:
                            ck += 1
                            if ck >= 2:
                                cj += 1
                            continue

                    # case 2
                    # bi_xmin ~= bk_xmax
                    # bi is side, smaller than bk
                    #elif 0.98 < (bi[0]/bk.selectNode(0).data[2]) and ((bk.selectNode(0).data[2]-bk.selectNode(0).data[0])*(bk.selectNode(0).data[3]-bk.selectNode(0).data[1])) > ((bi[2]-bi[0])*(bi[3]-bi[1])) and bk.selectNode(0).data[3] > bi[3]:
                    elif 0.98 < (bi[0]/bk.selectNode(0).data[2]) and 1.1 > (bk.selectNode(0).data[2]/bi[0]) and (bk_area) > (bi_area) and bk.selectNode(0).data[3] > bi[3] and bk.selectNode(0).data[0] < bi[0] and bk.selectNode(0).data[1] < bi[1]:
                        if debug == True:
                            print('case 2')
                        if ck != 0:
                            ck += 1
                        if flag == 0:
                            if len(q) > jdx and bi[0] > q[jdx].selectNode(0).data[2]:
                                p.append(q.pop(0))
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                break
                            elif int(bk.size()) > 2:
                                p.append(q.pop(0))
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                break
                            else:
                                if bi[5] != 8:
                                    q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                    break
                                else:
                                    bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                    break
                        elif flag == 1:
                            bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            flag = 0
                            break
                    # case 3
                    # continue
                    elif iou == 0.0:
                        if ck != 0:
                            ck += 1
                        if debug == True:
                            print('case 3')
                        if jdx == (len(q)-1) or len(q) == 1:
                            q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                            break
                        else:
                            continue
                    # case 4
                    elif iou > 0.0:
                        if ck != 0:
                            ck += 1
                        if debug == True:
                            print('case 4')
                        # a)
                        if bk.selectNode(0).data[0] < bi[0] and bk.selectNode(0).data[1] < bi[1] and bk.selectNode(0).data[2] > bi[2] and bk.selectNode(0).data[3] > bi[3]:
                            if debug == True:
                                print('1')
                            if bi[5] == 8 or bi[5] == 9:
                                if bi[4] > 0.6 or flag2 == 1:
                                    bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                    flag2 = 0
                                    break
                                elif bk.selectNode(0).data[5] == 0 and len(q) > 1: # 종이 일 때 q 안에 물체가 두개 이상일 때,
                                    flag2 = 1
                                    continue
                                elif bi[5] == 9:
                                    bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                    flag2 = 0
                                    break
                                else:
                                    q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                    break
                            else:
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                break
                            # insert bi into L(bk)
                            #break
                        # b-1)
                        elif 0.98 < (min(bk.selectNode(0).data[0],bi[0])/(max(bk.selectNode(0).data[0],bi[0])+1e-6)) and 0.98 < (min(bk.selectNode(0).data[3],bi[3])/(max(bk.selectNode(0).data[3],bi[3])+1e-6)):
                            if debug == True:
                                print('2')
                            if (bk_area) > (bi_area):
                                if bi[5] == 8 or bi[5] == 9:
                                    bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]) # 이번만
                                    #q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[5]]))
                                else:
                                    q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                            else:
                                bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            break
                        # b-2)
                        elif 0.98 < (min(bk.selectNode(0).data[2],bi[2])/(max(bk.selectNode(0).data[2],bi[2])+1e-6)) and 0.98 < (min(bk.selectNode(0).data[3],bi[3])/(max(bk.selectNode(0).data[3],bi[3])+1e-6)):
                            if debug == True:
                                print('3')
                            if (bk_area) > (bi_area):
                                if int(bk.size()) == 3:
                                    q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                else:
                                    bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            else:
                                if int(bk.size()) == 3:
                                    q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                else:
                                    bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            break
                        # b-3)
                        elif 0.98 < (min(bk.selectNode(0).data[1],bi[1])/(max(bk.selectNode(0).data[1],bi[1])+1e-6)) and 0.90 < (min(bk.selectNode(0).data[2],bi[2])/(max(bk.selectNode(0).data[2],bi[2])+1e-6)) and 0.95 < (min(bk.selectNode(0).data[0],bi[0])/(max(bk.selectNode(0).data[0],bi[0])+1e-6)): # 0.98 -> 0.90
                            if debug == True:
                                print('4')
                            if (bk_area) > (bi_area):
                                if bi[5] == 8 or bi[5] == 9:
                                    bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]) #이번만
                                    #q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[5]]))
                                else:
                                    q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                            else:
                                if int(bk.selectNode(0).data[5]) == 8 or int(bk.selectNode(0).data[5]) == 9:
                                    bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                else:
                                    q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                            break
                        # d) bk is side + bi is bigger than bk
                        elif 0.98 < (bk.selectNode(0).data[2]/(bi[0]+1e-6)) and bk.selectNode(0).data[1] > bi[1] and bk.selectNode(0).data[3] > bi[3] and bk.selectNode(0).data[1] < bi[1] and (bk_area) < (bi_area):
                            if debug == True:
                                print('5')
                            bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            break
                        elif 0.98 < (min(bk.selectNode(0).data[0],bi[0])/(max(bk.selectNode(0).data[0],bi[0])+1e-6)) and bi_area > bk_area and 0.98 < iou/((bk_area/bi_area)+1e-6):
                            if debug == True:
                                print('6')
                            bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            break
                        elif bi_area > bk_area and bk.selectNode(0).data[1] > bi[1] and bk.selectNode(0).data[3] < bi[3]:
                            if debug == True:
                                print('7')
                            if flag == 0:
                                if bidx == len(bounding_box)-1 or len(bounding_box) <= 3:
                                    if 0.95 < (min(bk.selectNode(0).data[0],bi[0])/(max(bk.selectNode(0).data[0],bi[0])+1e-6)):
                                        #print('check')
                                        if int(bk.selectNode(0).data[5]) != 8:
                                            q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                        else:
                                            bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                        break
                                    else:
                                        #print('check2')
                                        q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                        break
                                else:
                                    if int(bk.selectNode(0).data[5]) == 8 and bi[5] != 8:
                                        bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                        break
                                    else:
                                        q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                        flag += 1
                                        break
                            elif flag == 1:
                                bk.insertFirst([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                                flag = 0
                                break
                        # 아래쪽 side
                        elif bi_area < bk_area and 0.98 < (bi[1]/(bk.selectNode(0).data[3]+1e-6)) and bk.selectNode(0).data[0] < bi[0] and bk.selectNode(0).data[2] > bi[2]:
                            if debug == True:
                                print('8')
                            if bi[5] != 8:
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                            else:
                                bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            break
                        # inside 지만, 약간 튀어 나온 inside
                        elif bi_area < bk_area and 0.95 < iou/((bi_area/bk_area)+1e-6) and 0.98 < (bk.selectNode(0).data[1]/(bi[1]+1e-6)) and 0.98 < (bk.selectNode(0).data[3]/(bi[3]+1e-6)):
                            if debug == True:
                                print('9')
                            if bi[5] == 8 or bi[5] == 9:
                                bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            else:
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                            break
                        # 위쪽 side
                        elif bi_area < bk_area and 0.97 < (bk.selectNode(0).data[1]/(bi[3]+1e-6)) and bi[0] > bk.selectNode(0).data[0] and bi[3] < bk.selectNode(0).data[3]:
                            if debug == True:
                                print('10')
                            if bi[5] == 8:
                                bk.insertLast([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]])
                            else:
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                        else:
                            if debug == True:
                                print('last')
                            if jdx == (len(q)-1):
                                q.append(SingleLinkedList([bi[0],bi[1],bi[2],bi[3],bi[4],bi[5]]))
                                #cj += 1
                                continue
                            else:
                                continue

        for idxy in range(0, len(q)):
            p.append(q.pop(0))

        return p

    def get_iou(self, a, b, epsilon=1e-5):
        # COORDINATES OF THE INTERSECTION BOX
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)

        # handle case where there is NO overlap
        if (width<0) or (height <0):
            return 0.0
        area_overlap = width * height
        # COMBINED AREA
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined+epsilon)
        return iou

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __str__(self):
        return str(self.data)

class SingleLinkedList:
    def __init__(self, data):
        new_node = Node(data)
        self.head = new_node
        self.list_size = 1

    def __str__(self):
        print_list = '[ '
        node = self.head
        while True:
            print_list += str(node)
            if node.next == None:
                break
            node = node.next
            print_list += ', '
        print_list += ' ]'
        return print_list

    def insertFirst(self, data):
        new_node = Node(data)
        temp_node = self.head
        self.head = new_node
        self.head.next = temp_node
        self.list_size += 1

    def insertLast(self, data):
        node = self.head
        while True:
            if node.next == None:
                break
            node = node.next

        new_node = Node(data)
        node.next = new_node
        self.list_size += 1

    def insertMiddle(self, num, data):
        if self.head.next == None:
            self.insertLast(data)
            return
        node = self.selectNode(num)
        new_node = Node(data)
        temp_next = node.next
        node.next = new_node
        new_node.next = temp_next
        self.list_size += 1

    def selectNode(self, num):
        if self.list_size < num:
            print("Overflow")
            return
        node = self.head
        count = 0
        while count < num:
            node = node.next
            count += 1
        return node

    def deleteNode(self, num):
        if self.list_size < 1:
            return # Underflow
        elif self.list_size < num:
            return # Overflow

        if num == 0:
            self.deleteHead()
            return
        node = self.selectNode(num - 1)
        node.next = node.next.next
        del_node = node.next
        del del_node

    def deleteHead(self):
        node = self.head
        self.head = node.next
        del node

    def size(self):
        return str(self.list_size)

