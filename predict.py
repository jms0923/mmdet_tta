from os import path, listdir
from argparse import ArgumentParser
import json
from tqdm import tqdm
import re

from torchvision.transforms.functional import center_crop, hflip
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# from ELC import inference as elc
from predict_utils import PostProcessor
from PIL import Image
import numpy as np


NO_RESULT_CLASS = 6 # plastic
STANDARD_HEIGHT = 720
STANDARD_WIDTH = 1280
CENTER_CROP_RATIO = 0.8


def makeImgList(dir):
    imgFileNames = listdir(dir)
    imgFileNames = [x for x in imgFileNames if x.endswith(('jpg', 'JPG', 'jpeg', 'JPEG'))]
    imgFileNames = [path.join(dir, x) for x in imgFileNames]

    return imgFileNames

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def savePathFromImgPath(save_dir, imgPath):
    return path.join(save_dir, imgPath.split('/')[-1])

def _pillow2array(img, flag='color', channel_order='bgr'):
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag == 'color':
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag == 'grayscale':
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale" or "unchanged", '
                f'but got {flag}')
    return array

def imgPairSingle(imgPath, hflipBool=False):
    img = Image.open(imgPath)
    if img.width < img.height:
        img = img.transpose(Image.ROTATE_90)
    if hflipBool:
        img = hflip(img)
    # if centerCrop or (img.width > STANDARD_WIDTH and img.height > STANDARD_HEIGHT):
    #     cropHeight = int(img.height * CENTER_CROP_RATIO)
    #     cropWidth = int(img.width * CENTER_CROP_RATIO)
    #     img = center_crop(img, (cropHeight, cropWidth))
    img = _pillow2array(img, flag='color', channel_order='bgr')
    
    return  imgPath, img

def imgPairList(imgFileNames):
    for imgPath in imgFileNames:
        yield imgPairSingle(imgPath)

def inference(detectorsModel, detectorsPostProcessor, KEY_OBJECT, imgList, SCORE_CHECKER, SAVE_DIR):
    if SCORE_CHECKER:
        f = open("no_key_object.csv", 'w')
        f.write("file_name,c1,c2,c3,c4,c5,c6,c7 \n")

    labelSum = [0] * 8
    nonResultImgPath = []
    for imgPath, img in tqdm(imgPairList(imgList)):
        result = inference_detector(detectorsModel, img)
        if not KEY_OBJECT:
            # detectorsPostProcessor.saveResult(img, result, show=False, out_file=savePathFromImgPath(SAVE_DIR, imgPath))
            nowLabels = detectorsPostProcessor.saveIitp(img, imgPath, result)
            if isinstance(nowLabels, list):
                # case : true => labels list
                for label in nowLabels:
                    labelSum[label] += 1
            else:
                # case : false
                imgPath, img = imgPairSingle(imgPath, hflipBool=True)
                result = inference_detector(detectorsModel, img)
                nowLabels = detectorsPostProcessor.saveIitp(img, imgPath, result)
                if isinstance(nowLabels, list):
                    # case : true => labels list
                    for label in nowLabels:
                        labelSum[label] += 1
                else:
                    nonResultImgPath.append(imgPath)
            if SCORE_CHECKER:
                # our score checker
                _, labels = detectorsPostProcessor.cropBoxes(img, result, out_file=None)
                output_class = [0] * 8
                f.write(imgPath.split("/")[-1])
                for label in labels:
                    output_class[label] = 1
                output_class.pop(0)
                for i in output_class:
                    f.write("," + str(i))
                f.write("," + "\n")
        else:
            label_class = [0, 1, 1, 2, 3, 4, 5, 6, 5, 6]
            output_class = [0, 0, 0, 0, 0, 0, 0]
            keypointLabes = []
            keypointBoxes = []
            labels, p = detectorsPostProcessor.get_key_object(imgPath, result, out_file=None)
            for idxx in p:
                if int(idxx.size()) > 1:
                    for idxxx in range(0, int(idxx.size())):
                        if idxx.selectNode(idxxx).data[5] == 8:
                            if 3 in idxx.selectNode(0).data or 4 in idxx.selectNode(0).data:
                                idxx.selectNode(idxxx).data[5] = 3
                        elif idxx.selectNode(idxxx).data[5] == 9:
                            if 3 in idxx.selectNode(0).data:
                                idxx.selectNode(idxxx).data[5] = 3
                            elif 4 in idxx.selectNode(0).data:
                                idxx.selectNode(idxxx).data[5] = 0
                        label = label_class[idxx.selectNode(idxxx).data[5]]
                        keypointLabes.append(label+1)
                        keypointBoxes.append([int(idxx.selectNode(idxxx).data[0]), int(idxx.selectNode(idxxx).data[1]), int(idxx.selectNode(idxxx).data[2]), int(idxx.selectNode(idxxx).data[3])])
                        output_class[label] = 1
                        detectorsPostProcessor.annoMaker(imgPath, keypointBoxes, keypointLabes, labelChanger=False)
                elif int(idxx.size()) == 1:
                    for i in labels:
                        if i == 8:
                            if 3 in labels or 4 in labels:
                                i = 3
                        if i == 9:
                            if 3 in labels:
                                i = 3
                            elif 4 in labels:
                                i = 0
                        label = label_class[i]
                        output_class[label] = 1
                        detectorsPostProcessor.annoMaker(imgPath, [[100,200,300,400]], [label+1], labelChanger=False)
                else:
                    print('no result twice')
                    # no results
                    # output_class[NO_RESULT_CLASS-1] = 1
                    imgPath, img = imgPairSingle(imgPath, hflipBool=True)
                    result = inference_detector(detectorsModel, img)

                    output_class = [0, 0, 0, 0, 0, 0, 0]
                    keypointLabes = []
                    keypointBoxes = []
                    labels, p = detectorsPostProcessor.get_key_object(imgPath, result, out_file=None)
                    for idxx in p:
                        if int(idxx.size()) > 1:
                            for idxxx in range(0, int(idxx.size())):
                                if idxx.selectNode(idxxx).data[5] == 8:
                                    if 3 in idxx.selectNode(0).data or 4 in idxx.selectNode(0).data:
                                        idxx.selectNode(idxxx).data[5] = 3
                                elif idxx.selectNode(idxxx).data[5] == 9:
                                    if 3 in idxx.selectNode(0).data:
                                        idxx.selectNode(idxxx).data[5] = 3
                                    elif 4 in idxx.selectNode(0).data:
                                        idxx.selectNode(idxxx).data[5] = 0
                                label = label_class[idxx.selectNode(idxxx).data[5]]
                                keypointLabes.append(label+1)
                                keypointBoxes.append([int(idxx.selectNode(idxxx).data[0]), int(idxx.selectNode(idxxx).data[1]), int(idxx.selectNode(idxxx).data[2]), int(idxx.selectNode(idxxx).data[3])])
                                output_class[label] = 1
                                detectorsPostProcessor.annoMaker(imgPath, keypointBoxes, keypointLabes, labelChanger=False)
                        elif int(idxx.size()) == 1:
                            for i in labels:
                                if i == 8:
                                    if 3 in labels or 4 in labels:
                                        i = 3
                                if i == 9:
                                    if 3 in labels:
                                        i = 3
                                    elif 4 in labels:
                                        i = 0
                                label = label_class[i]
                                output_class[label] = 1
                                detectorsPostProcessor.annoMaker(imgPath, [[100,200,300,400]], [label+1], labelChanger=False)
                        else:
                            output_class[NO_RESULT_CLASS-1] = 1
                            nonResultImgPath.append(imgPath)

            if SCORE_CHECKER:
                # our score checker
                f.write(imgPath.split("/")[-1])
                for i in output_class:
                    f.write("," + str(i))
                f.write("," + "\n")

    for imgPath in nonResultImgPath:
        detectorsPostProcessor.annoMaker(imgPath, [[100,200,300,400]], [NO_RESULT_CLASS], labelChanger=False)

    with open('./t3_res_0022.json', 'w') as jsonFile:
        json.dump(detectorsPostProcessor.iitpJson, jsonFile)

def main():
    # DetectoRS options
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image files path')
    parser.add_argument('--config', default='/aichallenge/config.py')
    parser.add_argument('--checkpoint', default='/aichallenge/epoch.pth')
    # parser.add_argument('--config', default="/home/ubuntu/minseok/mmdetection/work_dirs/20201205_detectors_r50_seperateCanLabel_MST/2020_1205_detectors_cascade_rcnn_r50_1x_coco_MST.py")
    # parser.add_argument('--checkpoint', default="/home/ubuntu/minseok/mmdetection/work_dirs/20201205_detectors_r50_seperateCanLabel_MST/only_weights_epoch_26.pth")

    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--threshold', default=0.46)
    parser.add_argument('--score_checker', default=True)
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--key_object', default=False)
    args = parser.parse_args()

    # load image list
    imgList = sorted_aphanumeric(makeImgList(args.img_dir))
    # build DetectoRS
    detectorsModel = init_detector(args.config, args.checkpoint, device=args.devices)
    # detectorsModel.CLASSES = ('paper', 'paperpack', 'papercup', 'can', 'bottle', 'pet', 'plastic', 'vinyl', 'cap', 'label')
    detectorsModel.CLASSES = ('paper', 'paperpack', 'papercup', 'can', 'bottle', 'pet', 'plastic', 'vinyl', 'cap_can', 'cap_plastic', 'label_paper', 'label_vinyl')
    print('detectorsModel.CLASSES : ', detectorsModel.CLASSES)
    detectorsPostProcessor = PostProcessor(detectorsModel.CLASSES, score_thr=args.threshold)

    inference(detectorsModel, detectorsPostProcessor, args.key_object, imgList, args.score_checker, args.save_dir)


if __name__ == '__main__':
    main()
