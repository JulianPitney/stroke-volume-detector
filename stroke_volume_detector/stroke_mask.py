from skimage import io
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.applications.xception import preprocess_input
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
import copy
import argparse
from time import sleep


LABEL_DICT = {'background': 0, 'normal': 1, 'stroke': 2}
COLOR_DICT = {'background': (0, 0, 0), 'normal': (0, 255, 0), 'stroke': (255, 0, 0)}


def ConvAndBatch(x, n_filters=64, kernel=(2, 2), strides=(1, 1), padding='valid', activation='relu'):
    filters = n_filters

    conv_ = Conv2D(filters=filters,
                   kernel_size=kernel,
                   strides=strides,
                   padding=padding)

    batch_norm = BatchNormalization()

    activation = Activation(activation)

    x = conv_(x)
    x = batch_norm(x)
    x = activation(x)

    return x


def ConvAndAct(x, n_filters, kernel=(1, 1), activation='relu', pooling=False):
    poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')
    convLayer = Conv2D(filters=n_filters,
                       kernel_size=kernel,
                       strides=1)

    activation = Activation(activation)

    if pooling:
        x = poolingLayer(x)

    x = convLayer(x)
    x = activation(x)

    return x


def AttentionRefinmentModule(inputs, n_filters):
    filters = n_filters

    poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')

    x = poolingLayer(inputs)
    x = ConvAndBatch(x, kernel=(1, 1), n_filters=filters, activation='sigmoid')

    return multiply([inputs, x])


def FeatureFusionModule(input_f, input_s, n_filters):
    concatenate = Concatenate(axis=-1)([input_f, input_s])

    branch0 = ConvAndBatch(concatenate, n_filters=n_filters, kernel=(3, 3), padding='same')
    branch_1 = ConvAndAct(branch0, n_filters=n_filters, pooling=True, activation='relu')
    branch_1 = ConvAndAct(branch_1, n_filters=n_filters, pooling=False, activation='sigmoid')

    x = multiply([branch0, branch_1])
    return Add()([branch0, x])


def ContextPath(layer_13, layer_14):
    globalmax = GlobalAveragePooling2D()

    block1 = AttentionRefinmentModule(layer_13, n_filters=1024)
    block2 = AttentionRefinmentModule(layer_14, n_filters=2048)

    global_channels = globalmax(block2)
    block2_scaled = multiply([global_channels, block2])

    block1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(block1)
    block2_scaled = UpSampling2D(size=(4, 4), interpolation='bilinear')(block2_scaled)

    cnc = Concatenate(axis=-1)([block1, block2_scaled])

    return cnc


def FinalModel(x, layer_13, layer_14):
    x = ConvAndBatch(x, 32, strides=2)
    x = ConvAndBatch(x, 64, strides=2)
    x = ConvAndBatch(x, 156, strides=2)

    # context path
    cp = ContextPath(layer_13, layer_14)
    fusion = FeatureFusionModule(cp, x, 3)
    ans = UpSampling2D(size=(8, 8), interpolation='bilinear')(fusion)

    return ans


def get_model(imageNet=False):
    inputs = Input(shape=(608, 608, 3))
    x = Lambda(lambda image: preprocess_input(image))(inputs)
    if imageNet:
        xception = Xception(weights='imagenet', input_shape=(608, 608, 3), include_top=False)
    else:
        xception = Xception(weights=None, input_shape=(608, 608, 3), include_top=False)

    tail_prev = xception.get_layer('block13_pool').output
    tail = xception.output

    output = FinalModel(x, tail_prev, tail)

    return inputs, xception.input, output

def readTif(tifPath, keepThreshold=100, imgShape=(608, 608), filterDark=False, returnOriginal=False):
    assert os.path.exists(tifPath)
    imgStack = io.imread(tifPath)
    (steps, height, width) = imgStack.shape

    processedStacks = []
    originalStacks = []
    for step in range(steps):
        img8 = cv2.normalize(imgStack[step], None, 0, 255, cv2.NORM_MINMAX)
        img8 = np.asarray(img8, dtype='uint8')
        blur = cv2.GaussianBlur(img8, (3, 3), 0)
        imgResize = cv2.resize(blur, imgShape)
        img3Channel = cv2.cvtColor(imgResize, cv2.COLOR_GRAY2RGB)
        if filterDark:
            if np.max(imgResize) >= keepThreshold:
                processedStacks.append(img3Channel)
        else:
            processedStacks.append(img3Channel)
        if returnOriginal:
            imgOrignal = cv2.normalize(imgStack[step], None, 0, 255, cv2.NORM_MINMAX)
            imgOrignal = np.asarray(imgOrignal, dtype='uint8')
            originalStacks.append(imgOrignal)

    processedStacks = np.array(processedStacks)

    if returnOriginal:
        return processedStacks, originalStacks
    else:
        return processedStacks, imgStack.shape

def predict2Mask(prediction):
    copyMask = np.zeros(shape=(prediction.shape[0], prediction.shape[1], 3), dtype='uint8')
    binarayMask = np.argmax(prediction, axis=-1)
    for key in LABEL_DICT.keys():
        label = np.zeros(shape=(len(LABEL_DICT.keys()), ), dtype='uint8')
        label[LABEL_DICT[key]] = 1
        copyMask[binarayMask == LABEL_DICT[key], :] = label
    return copyMask

def label2Color(labelMask):
    copyMask = copy.deepcopy(labelMask)
    canvas = np.zeros(shape=(copyMask.shape[0], copyMask.shape[1], 3), dtype='uint8')
    for key in LABEL_DICT.keys():
        canvas[copyMask[:, :, LABEL_DICT[key]] == 1, :] = COLOR_DICT[key]
    return canvas

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

class STROKE_MASK:
    def __init__(self):

        weights_dir = os.path.dirname(os.path.realpath(__file__)) + "\\..\\weights\\"
        inputs, xception_inputs, ans = get_model()
        self.model = Model(inputs=[inputs, xception_inputs], outputs=[ans])
        self.model_weights = os.path.join(weights_dir,
                                          [item for item in os.listdir(weights_dir) if ".index" in item][0].replace(
                                              ".index", ""))
        self.model.load_weights(self.model_weights)

    def visualize(self):
        """
        Visualizing the model structure
        :return: None
        """
        self.model.summary()

    def predict(self, tif_stack_list, original_shape, color=False):
        """
        :param tif_stack_list: A list of numpy arrays, each array is a stack.
        :param heatmap: if true, Return the original predictions, has the same shape as input.
                        axis 0 => background, 1 => Normal, 2 => Stroke
                        The number in each axis indicates the confidence value.
                        [
                            (steps, height, width, 3),
                            (steps, height, width, 3),
                            ...
                        ]
                        If False, Return the one-hot label of predictions.
                        [
                            (steps, height, width, 3), [1, 0, 0] => background, [0, 1, 0] => Normal, [0, 0, 1] => Stroke
                            (steps, height, width, 3),
                            ...
                        ]
        :param color: Covert one-hot predictions to color mask.
        :return: A list of predictions
        """
        imgShape = (608, 608)
        result = []
        while(len(tif_stack_list)) > 0:
            tif_stack = tif_stack_list.pop()
            assert len(tif_stack.shape) >= 3
            steps = tif_stack.shape[0]
            processedStack = []
            for step in range(steps):
                img8 = cv2.normalize(tif_stack[step], None, 0, 255, cv2.NORM_MINMAX)
                img8 = np.asarray(img8, dtype='uint8')
                blur = cv2.GaussianBlur(img8, (3, 3), 0)
                imgResize = cv2.resize(blur, imgShape)
                if len(imgResize.shape) == 3:
                    img3Channel = imgResize
                else:
                    img3Channel = cv2.cvtColor(imgResize, cv2.COLOR_GRAY2RGB)

                processedStack.append(img3Channel)
            del tif_stack
            processedStack = np.asarray(processedStack, dtype=np.float32)
            x = [processedStack, processedStack]
            predictions = self.model.predict(x, batch_size=4, verbose=1)
            del processedStack
            del x

            predictions = list(predictions)
            slices = []

            for i in range(0, len(predictions)):
                prediction = predictions[i]
                if color:
                    temp = label2Color(predict2Mask(prediction))
                else:
                    temp = predict2Mask(prediction)
                slices.append(temp[:, :, 0])
            slices = np.array(slices, dtype=np.uint8)

            result.append(slices)

        return result

def main():

    parser = argparse.ArgumentParser(description='This tool takes a directory of tiff stacks as input. It will place each pixel into one of three categories; (0=Background, 1=Normal, 2=Stroke) It will store these results in an image mask of the same size and shape as the input stack.')
    parser.add_argument('--scans_dir', metavar='STITCHED_SCANS_DIR', dest='STITCHED_SCANS_DIR', action='store', required=True, help='Full path to directory where stitched lightsheet scans are. This directory should ONLY contain the stitched lightsheet scans.')
    args = vars(parser.parse_args())

    STITCHED_SCANS_DIR = args.get('STITCHED_SCANS_DIR')
    if not os.path.isdir(STITCHED_SCANS_DIR):
        print(STITCHED_SCANS_DIR + " does not exist. Exiting.")



    scans = os.listdir(path=STITCHED_SCANS_DIR)
    for scan in scans:
        if os.path.isfile(STITCHED_SCANS_DIR + "\\" + scan) and scan[-4:] == '.tif':
            continue
        else:
            print(STITCHED_SCANS_DIR + "\\" + scan + " does not exist or does not have the .tif extension. Skipping file.")
            scans.remove(scan)

    S = STROKE_MASK()

    for scan in scans:
        print("Loading " + scan)
        stack, original_shape = readTif(STITCHED_SCANS_DIR + "\\" + scan, imgShape=(608, 608))
        result = S.predict([stack], original_shape, color=True)
        del stack
        print("Saving " + STITCHED_SCANS_DIR + "\\" + scan[:-4] + "_stroke_mask.tif")
        io.imsave(STITCHED_SCANS_DIR + "\\" + scan[:-4] + "_stroke_mask.tif", result[0])
        del result

if __name__ == '__main__':
    main()

