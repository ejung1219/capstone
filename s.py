from skimage import io, transform
import skimage
import matplotlib.patches as patches
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import imutils
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def video_partition():
    count = []
    for i in range(1, 2):
        filepath = './video1.mp4'
        if os.path.isfile(filepath):
            video = cv2.VideoCapture(filepath)

            if not video.isOpened():
                print("Could not Open :", filepath)
                exit(0)

            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video.get(cv2.CAP_PROP_FPS)

            cnt = 0

            while(video.isOpened()):
                if (cnt + 1) * (int)(fps) > length:
                    break
                ret, image = video.read()
                if(int(video.get(1)) % (int)(fps) == 0):
                    createFolder("./video%d" % i)
                    cv2.imwrite(
                        "./video%d/frame%d.png" % (i, cnt), image)

                    cnt += 1
            video.release()
        else:
            break
        count.append(cnt)

    return count, i - 1


def load_image(image_path):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    img = skimage.img_as_float(io.imread(image_path))
    if len(img.shape) == 2:
        img = np.array([img, img, img]).swapaxes(0, 2)
    return img


def rescale(img, input_height, input_width):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    aspect = img.shape[1] / float(img.shape[0])
    if (aspect > 1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = transform.resize(img, (input_width, res))
    if (aspect < 1):
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = transform.resize(img, (res, input_height))
    if (aspect == 1):
        imgScaled = transform.resize(img, (input_width, input_height))
    return imgScaled


def crop_center(img, cropx, cropy):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def normalize(img, mean=128, std=128):
    img = (img * 256 - mean) / std
    return img


def prepare(img_uri):
    img = load_image(img_uri)
    img = rescale(img, 300, 300)
    img = transform.resize(img, (300, 300))
    img = normalize(img)
    return img


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error:Creating directory.' + directory)


def prepare_tensor(inputs, fp16=False):
    NHWC = np.array(inputs)
    NCHW = np.swapaxes(np.swapaxes(NHWC, 1, 3), 2, 3)
    tensor = torch.from_numpy(NCHW)
    tensor = tensor.contiguous()
    #tensor = tensor.cuda()
    tensor = tensor.float()
    if fp16:
        tensor = tensor.half()
    return tensor


def ssd():
    k, v = video_partition()
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    ssd_model = torch.load(
        './ssd.pth.tar', map_location=torch.device(device))
    utils = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    ssd_model.to('cpu')
    ssd_model.eval()

    target = cv2.imread("./target.png")
    print(v)
    for j in range(1, 2):
        count = 1
        uris = list()
        for i in range(k[j - 1]):
            uris.append("./video%d/frame%d.png" % (j, i))

        inputs = [prepare(uri) for uri in uris]
        tensor = prepare_tensor(inputs)

        with torch.no_grad():
            detections_batch = ssd_model(tensor)

        results_per_input = utils.decode_results(detections_batch)
        best_results_per_input = [utils.pick_best(
            results, 0.6) for results in results_per_input]
        classes_to_labels = utils.get_coco_object_dictionary()
        createFolder("./result%d" % j)
        cv2.imwrite("./target.png" % j, target)
        for image_idx in range(len(best_results_per_input)):
            tmp = plt.imread(
                './video%d/frame%d.png' % (j, image_idx))
            #fig, ax = plt.subplots(1)
            image = inputs[image_idx] / 2 + 0.5
            bboxes, classes, confidences = best_results_per_input[image_idx]
            for idx in range(len(bboxes)):
                left, bot, right, top = bboxes[idx]
                x, y, w, h = [val * 300 for val in [left,
                                                    bot, right - left, top - bot]]
                if classes_to_labels[classes[idx] - 1] == 'person':
                    tmp = rescale(tmp, 300, 300)
                    tmp = transform.resize(tmp, (300, 300))
                    if y < 0:
                        y = 0
                    if x < 0:
                        x = 0
                    cropped = tmp[int(y):int(y+h), int(x):int(x+w)]
                    plt.imsave("./result%d/%d.png" %
                               (j, count), cropped)
                    count += 1
                #rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                # ax.add_patch(rect)
                #ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
