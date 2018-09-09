import mxnet as mx
from mxnet import ndarray as F
import os
from skimage.transform import resize
from skimage.io import imsave
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet.gluon.model_zoo import vision

path = '/home/td/Documents/radio/'

batch_size = 1
img_rows = 256
img_cols = 256
data_shape = (batch_size, 1, img_rows, img_cols)

def dice_coef(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=(1, 2, 3))
    return mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.sum(y_true, axis=(1, 2, 3)) + mx.sym.sum(y_pred, axis=(1, 2, 3)) + 1.))


def dice_coef_loss(y_true, y_pred):
    intersection = mx.sym.sum(mx.sym.broadcast_mul(y_true, y_pred), axis=1, )
    return -mx.sym.broadcast_div((2. * intersection + 1.),(mx.sym.broadcast_add(mx.sym.sum(y_true, axis=1), mx.sym.sum(y_pred, axis=1)) + 1.))


def build_unet():
    data = mx.sym.Variable(name='data')
    label = mx.sym.Variable(name='label')
    conv1 = mx.sym.Convolution(data, num_filter=32, kernel=(3, 3), pad=(1, 1), name='conv1_1')
    conv1 = mx.sym.BatchNorm(conv1, name='bn1_1')
    conv1 = mx.sym.Activation(conv1, act_type='relu', name='relu1_1')
    conv1 = mx.sym.Convolution(conv1, num_filter=32, kernel=(3, 3), pad=(1, 1), name='conv1_2')
    conv1 = mx.sym.BatchNorm(conv1, name='bn1_2')
    conv1 = mx.sym.Activation(conv1, act_type='relu', name='relu1_2')
    pool1 = mx.sym.Pooling(conv1, kernel=(2, 2), pool_type='max', name='pool1')

    conv2 = mx.sym.Convolution(pool1, num_filter=64, kernel=(3, 3), pad=(1, 1), name='conv2_1')
    conv2 = mx.sym.BatchNorm(conv2, name='bn2_1')
    conv2 = mx.sym.Activation(conv2, act_type='relu', name='relu2_1')
    conv2 = mx.sym.Convolution(conv2, num_filter=64, kernel=(3, 3), pad=(1, 1), name='conv2_2')
    conv2 = mx.sym.BatchNorm(conv2, name='bn2_2')
    conv2 = mx.sym.Activation(conv2, act_type='relu', name='relu2_2')
    pool2 = mx.sym.Pooling(conv2, kernel=(2, 2), pool_type='max', name='pool2')

    conv3 = mx.sym.Convolution(pool2, num_filter=128, kernel=(3, 3), pad=(1, 1), name='conv3_1')
    conv3 = mx.sym.BatchNorm(conv3, name='bn3_1')
    conv3 = mx.sym.Activation(conv3, act_type='relu', name='relu3_1')
    conv3 = mx.sym.Convolution(conv3, num_filter=128, kernel=(3, 3), pad=(1, 1), name='conv3_2')
    conv3 = mx.sym.BatchNorm(conv3, name='bn3_2')
    conv3 = mx.sym.Activation(conv3, act_type='relu', name='relu3_2')
    pool3 = mx.sym.Pooling(conv3, kernel=(2, 2), pool_type='max', name='pool3')

    conv4 = mx.sym.Convolution(pool3, num_filter=256, kernel=(3, 3), pad=(1, 1), name='conv4_1')
    conv4 = mx.sym.BatchNorm(conv4, name='bn4_1')
    conv4 = mx.sym.Activation(conv4, act_type='relu', name='relu4_1')
    conv4 = mx.sym.Convolution(conv4, num_filter=256, kernel=(3, 3), pad=(1, 1), name='conv4_2')
    conv4 = mx.sym.BatchNorm(conv4, name='bn4_2')
    conv4 = mx.sym.Activation(conv4, act_type='relu', name='relu4_2')
    pool4 = mx.sym.Pooling(conv4, kernel=(2, 2), pool_type='max', name='pool4')

    conv5 = mx.sym.Convolution(pool4, num_filter=512, kernel=(3, 3), pad=(1, 1), name='conv5_1')
    conv5 = mx.sym.BatchNorm(conv5, name='bn5_1')
    conv5 = mx.sym.Activation(conv5, act_type='relu', name='relu5_1')
    conv5 = mx.sym.Convolution(conv5, num_filter=512, kernel=(3, 3), pad=(1, 1), name='conv5_2')
    conv5 = mx.sym.BatchNorm(conv5, name='bn5_2')
    conv5 = mx.sym.Activation(conv5, act_type='relu', name='relu5_2')

    trans_conv6 = mx.sym.Deconvolution(conv5, num_filter=256, kernel=(2, 2), stride=(1, 1), no_bias=True,
                                       name='trans_conv6')
    up6 = mx.sym.concat(*[trans_conv6, conv4], dim=1, name='concat6')
    conv6 = mx.sym.Convolution(up6, num_filter=256, kernel=(3, 3), pad=(1, 1), name='conv6_1')
    conv6 = mx.sym.BatchNorm(conv6, name='bn6_1')
    conv6 = mx.sym.Activation(conv6, act_type='relu', name='relu6_1')
    conv6 = mx.sym.Convolution(conv6, num_filter=256, kernel=(3, 3), pad=(1, 1), name='conv6_2')
    conv6 = mx.sym.BatchNorm(conv6, name='bn6_2')
    conv6 = mx.sym.Activation(conv6, act_type='relu', name='relu6_2')

    trans_conv7 = mx.sym.Deconvolution(conv6, num_filter=128, kernel=(2, 2), stride=(1, 1), no_bias=True,
                                       name='trans_conv7')
    up7 = mx.sym.concat(*[trans_conv7, conv3], dim=1, name='concat7')
    conv7 = mx.sym.Convolution(up7, num_filter=128, kernel=(3, 3), pad=(1, 1), name='conv7_1')
    conv7 = mx.sym.BatchNorm(conv7, name='bn7_1')
    conv7 = mx.sym.Activation(conv7, act_type='relu', name='relu7_1')
    conv7 = mx.sym.Convolution(conv7, num_filter=128, kernel=(3, 3), pad=(1, 1), name='conv7_2')
    conv7 = mx.sym.BatchNorm(conv7, name='bn7_2')
    conv7 = mx.sym.Activation(conv7, act_type='relu', name='relu7_2')

    trans_conv8 = mx.sym.Deconvolution(conv7, num_filter=64, kernel=(2, 2), stride=(1, 1), no_bias=True, name='trans_conv8')
    up8 = mx.sym.concat(*[trans_conv8, conv2], dim=1, name='concat8')
    conv8 = mx.sym.Convolution(up8, num_filter=64, kernel=(3, 3), pad=(1, 1), name='conv8_1')
    conv8 = mx.sym.BatchNorm(conv8, name='bn8_1')
    conv8 = mx.sym.Activation(conv8, act_type='relu', name='relu8_1')
    conv8 = mx.sym.Convolution(conv8, num_filter=64, kernel=(3, 3), pad=(1, 1), name='conv8_2')
    conv8 = mx.sym.BatchNorm(conv8, name='bn8_2')
    conv8 = mx.sym.Activation(conv8, act_type='relu', name='relu8_2')

    trans_conv9 = mx.sym.Deconvolution(conv8, num_filter=32, kernel=(2, 2), stride=(1, 1), no_bias=True, name='trans_conv9')
    up9 = mx.sym.concat(*[trans_conv9, conv1], dim=1, name='concat9')
    conv9 = mx.sym.Convolution(up9, num_filter=32, kernel=(3, 3), pad=(1, 1), name='conv9_1')
    conv9 = mx.sym.BatchNorm(conv9, name='bn9_1')
    conv9 = mx.sym.Activation(conv9, act_type='relu', name='relu9_1')
    conv9 = mx.sym.Convolution(conv9, num_filter=32, kernel=(3, 3), pad=(1, 1), name='conv9_2')
    conv9 = mx.sym.BatchNorm(conv9, name='bn9_2')
    conv9 = mx.sym.Activation(conv9, act_type='relu', name='relu9_2')

    conv10 = mx.sym.Convolution(conv9, num_filter=1, kernel=(1, 1), name='conv10_1')
    conv10 = mx.sym.sigmoid(conv10, name='softmax')

    net = mx.sym.Flatten(conv10)
    loss = mx.sym.MakeLoss(dice_coef_loss(label, net), normalization='batch')
    mask_output = mx.sym.BlockGrad(conv10, 'mask')
    out = mx.sym.Group([loss, mask_output])
    #     return mx.sym.Custom(net, pos_grad_scale = pos, neg_grad_scale = neg, name = 'softmax', op_type = 'weighted_logistic_regression')
    #     return mx.sym.LogisticRegressionOutput(net, name='softmax')
    return out


def get_data():
    df = pd.read_csv(path + 'stage_1_train_labels.csv')

    patientids = list(set(df['patientId'].tolist()))

    images, masks = [], []

    # imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    # imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    for i in patientids:
        print(i)
        df_p = df[df['patientId'] == i]

        if df_p.shape[0] > 1 or df_p['Target'].tolist()[0] != 0:
            mask = np.zeros((1024, 1024))
            for k, v in df_p.iterrows():
                x = int(v['x'])
                y = int(v['y'])
                width = int(v['width'])
                height = int(v['height'])
                mask[x:x + width, y: y + height] = 1
                # mask[0,0] = 2
                # print(mask.max())
        else:
            continue

        np_image = pydicom.dcmread(path + 'stage_1_train_images/' + i + '.dcm').pixel_array
        np_image = np_image.astype(np.float64)
        np_image /= 255

        # np_image = imresize(np_image, (img_rows, img_cols))
        # mask = imresize(mask, (img_rows, img_cols))
        mask = (mask > .5).astype(int)

        for x_c in range(1024 // img_rows):
            for y_c in range(1024 // img_cols):
                np_image2 = np_image[x_c:x_c + 1024 // img_rows, y_c : y_c // img_cols]


                np_image = np.expand_dims(np_image, 0)
                mask = np.expand_dims(mask, 0)

                images.append(np_image)
                mask = mask.flatten()
                masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    train_x, val_x, train_y, val_y = train_test_split(images, masks, test_size=.05)
    print(train_x.shape)
    train_data = mx.io.NDArrayIter(train_x, train_y, batch_size, shuffle=True, label_name='label')
    val_data = mx.io.NDArrayIter(val_x, train_y, batch_size, label_name='label')
    return train_data, val_data


train_data, val_data = get_data()
net = build_unet()
ctx = [mx.gpu()]
unet = mx.mod.Module(net, context=ctx, data_names=('data',), label_names=('label',))


unet.bind(data_shapes=[['data', data_shape]], label_shapes=[['label', (batch_size, img_cols * img_rows)]])
unet.init_params(mx.initializer.Xavier(magnitude=6))
unet.init_optimizer(optimizer = 'adam',
                               optimizer_params=(
                                   ('learning_rate', 1E-4),
                                   ('beta1', 0.9),
                                   ('beta2', 0.99)
                              ))

epochs = 100
smoothing_constant  = .01
curr_losses = []
moving_losses = []
i = 0
best_val_loss = np.inf
for e in range(epochs):
    while True:
        try:
            batch = next(train_data)
        except StopIteration:
            train_data.reset()
            break
        unet.forward_backward(batch)
        loss = unet.get_outputs()[0]
        unet.update()
        curr_loss = F.mean(loss).asscalar()
        curr_losses.append(curr_loss)
        print(e, len(curr_losses), sum(curr_losses)/len(curr_losses), curr_loss)

        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                               else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        moving_losses.append(moving_loss)
        i += 1
    val_losses = []
    for batch in val_data:
        unet.forward(batch)
        loss = unet.get_outputs()[0]
        val_losses.append(F.mean(loss).asscalar())
        val_data.reset()
    val_loss = np.mean(val_losses)
    print(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        unet.save_checkpoint('best_unet', 0)
        print("Best model at Epoch %i" %(e+1))
    print("Epoch %i: Moving Training Loss %0.5f, Validation Loss %0.5f" % (e+1, moving_loss, val_loss))