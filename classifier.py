import pandas as pd
import tensorflow
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split
import glob
import torch
from torch.utils import data
import torch.utils.data.sampler
import tqdm
from modified_models import *
import pydicom
from scipy.ndimage import gaussian_gradient_magnitude, morphology

device = 'cuda'
path = '/home/td/Documents/radio/'


def generate_masks(img_name, df):
    df_copy = df[df['ImageId'] == img_name]

    image_sum  = np.zeros((768, 768))
    output = [1]
    for _, i in df_copy.iterrows():
        if len(i['EncodedPixels']) > 0:
            output = [0]

    return output



def load_image(path, df, mask = False):
    if mask:
        image_name = path.split('/')[-1].split('.')[0]
        df_copy = df[df['patientId'] == image_name]
        output = df_copy['Target'].tolist()[0]

        if output == 1:
            output = [1]
        else:
            output = [0]

        return torch.from_numpy(np.array(output))


    else:
        try:
            np_image = pydicom.dcmread(path).pixel_array
            np_image = np_image.astype(np.float64)
            np_image /= 255
            g1 = gaussian_gradient_magnitude(np_image, sigma=[.1, .9])
            g2 = gaussian_gradient_magnitude(np_image, sigma=[.9, .1])
            np_image = np.dstack((np.expand_dims(np_image, axis=2),
                             np.expand_dims(g1, axis=2),
                             np.expand_dims(g2, axis=2)))

        except:
            import traceback
            traceback.print_exc()
            np_image = np.zeros((1024, 1024, 3))
        return torch.from_numpy(np_image).float().permute([2, 0, 1])


class ShipDataset(data.Dataset):
    def __init__(self, file_list, is_test=False, is_val = False):
        self.is_test = is_test
        self.file_list = file_list
        self.len_multiplier = 1
        self.is_val = is_val

        df = pd.read_csv(path + 'stage_1_train_labels.csv')
        df = df.drop_duplicates(subset=['patientId'])
        print(df['Target'].mean())
        self.df = df.fillna('')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        image_path = self.file_list[index]

        if self.is_test:
            return image_path, load_image(image_path, self.df)

        mask = load_image(image_path, self.df, mask=True)
        image = load_image(image_path, self.df)
        return image, mask


def main():
    train_locs = glob.glob(path + 'stage_1_train_images/*')
    train_locs = [i for i in train_locs]

    train_locs, val_locs = train_test_split(train_locs, test_size=.001, random_state=1)
    print(len(train_locs), len(val_locs))
    # train_ds = ShipDataset(train_locs, is_val = True)
    val_ds = ShipDataset(val_locs, is_val = True)

    clf = resnet18().cuda()
    learning_rate = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)

    results = []
    patience = 5
    train_ds = ShipDataset(train_locs[:1000])

    for e in range(10000):
        train_loss = []


        for image, mask in tqdm.tqdm(data.DataLoader(train_ds, batch_size=1, shuffle=True)):
            image = image.type(torch.float).to(device)
            y_pred = clf(image)
            s_mask = mask.squeeze()
            s_y_pred = y_pred.squeeze()
            loss = loss_fn(y_pred, mask.to(device))
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        val_loss = []
        for image, mask in data.DataLoader(val_ds, batch_size=1, shuffle=False):
            image = image.to(device)
            y_pred = clf(image)
            s_mask = mask.squeeze()
            s_y_pred = y_pred.squeeze()

            print(mask, y_pred)

            loss = loss_fn(y_pred, mask.to(device))
            val_loss.append(loss.item())

        prev_val_loss = min(val_loss)
        results.append(prev_val_loss)
        print("Epoch: %d, Train: %.7f, Val: %.7f" % (e, min(train_loss), min(val_loss)))
        print(results.index(min(results)), len(results))
        if prev_val_loss == min(val_loss):
            torch.save(clf, path + 'classifier')

        if (results.index(min(results)) + patience) <= len(results):
            break


main()