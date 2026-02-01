from google.colab import drive
from google.colab.patches import cv2_imshow

import tensorflow as tf
import keras
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import *
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation
import tensorflow.keras.backend as K

import glob, os, cv2
import numpy as np
from tqdm import tqdm

drive.mount('/content/drive')

import zipfile

zip_path = "/content/drive/MyDrive/track_dataset.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    # 🔑 폴더 구분 없이 jpg만 필터
    jpg_files = [
        f for f in z.namelist()
        if f.lower().endswith('.jpg') and not f.endswith('/')
    ]

    print("📦 ZIP 내부 jpg 개수:", len(jpg_files))

    print("예시 파일명:")
    for name in jpg_files[:5]:
        print(name)

import os, glob, shutil, cv2
import numpy as np
from tqdm import tqdm

PATH = '/content/tmp'
ZIP_PATH = '/content/drive/MyDrive/track_dataset.zip'
EXTRACT_DIR = '/content/tmp/track_dataset_flat'

os.makedirs(PATH, exist_ok=True)

# ✅ 오염 방지: 추출 폴더 삭제
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# ✅ unzip
!unzip -o "$ZIP_PATH" -d "$EXTRACT_DIR" > /dev/null

# ✅ 핵심: 폴더가 있든 없든 전부 수집 (재귀)
file_list = sorted(glob.glob(f'{EXTRACT_DIR}/**/*.jpg', recursive=True))

print("✅ extracted jpg count:", len(file_list))
print("예시 경로:")
for p in file_list[:5]:
    print(p)

# ✅ 만약 0개면 unzip 결과 자체가 예상과 다름 → 바로 중단
assert len(file_list) > 0, "jpg를 하나도 못 찾음. ZIP 구조/압축해제 경로를 다시 확인해야 함."

label = []
img = []
bad = 0
none_img = 0
shape_bad = 0

for file in tqdm(file_list):
    base = os.path.basename(file)
    parts = os.path.splitext(base)[0].split('_')

    # label 파싱
    try:
        x = float(parts[0])
        y = float(parts[1])
    except:
        bad += 1
        continue

    X = cv2.imread(file)
    if X is None:
        none_img += 1
        continue

    # ✅ 크롭 안전장치: 해상도 달라서 잘리는 경우 방지
    if X.shape[0] < 270:
        shape_bad += 1
        continue

    img.append(X[120:270, :])
    label.append([x, y])

label = np.array(label, dtype=np.float32) / 400.0
img = np.array(img, dtype=np.uint8)

print("label parse fail:", bad)
print("imread fail:", none_img)
print("image too small fail:", shape_bad)
print("label/img shape:", label.shape, img.shape)

import random
idx = random.randint(0, len(img)-1)

print("label:", label[idx])
cv2_imshow(img[idx])

import matplotlib.pyplot as plt

plt.hist(label[:,0], bins=30)
plt.title("Label X distribution")
plt.show()

plt.hist(label[:,1], bins=30)
plt.title("Label Y distribution")
plt.show()


input1 = keras.layers.Input(shape=(150, 400, 3,))
x = keras.layers.Rescaling(1./255)(input1)
conv1 = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides= (2, 2), padding="same", activation="swish")(input1)
norm1 = keras.layers.BatchNormalization()(conv1)
pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)) (norm1)
conv2 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides= (2, 2), padding="same", activation="swish")(pool1)
norm2 = keras.layers.BatchNormalization()(conv2)
conv3 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides= (1, 1), padding="same", activation="swish")(norm2)
norm3 = keras.layers.BatchNormalization()(conv3)
add1 = keras.layers.Add()([norm2, norm3])
conv4 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides= (2, 2), padding="same", activation="swish")(add1)
norm4 = keras.layers.BatchNormalization()(conv4)
conv5 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides= (1, 1), padding="same", activation="swish")(norm4)
norm5 = keras.layers.BatchNormalization()(conv5)
add2 = keras.layers.Add()([norm4, norm5])
conv6 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides= (2, 2), padding="same", activation="swish")(add2)
norm6 = keras.layers.BatchNormalization()(conv6)
conv7 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides= (1, 1), padding="same", activation="swish")(norm6)
norm7 = keras.layers.BatchNormalization()(conv7)
add3 = keras.layers.Add()([norm6, norm7])
conv8 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides= (2, 2), padding="same", activation="swish")(add3)
norm7 = keras.layers.BatchNormalization()(conv8)
conv9 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides= (2, 2), padding="same", activation="swish")(norm7)
norm8 = keras.layers.BatchNormalization()(conv9)
flat1 = keras.layers.Flatten()(norm8)
dense1 = keras.layers.Dense(128, activation="swish")(flat1)
norm9 = keras.layers.BatchNormalization()(dense1)
dense2 = keras.layers.Dense(64, activation="swish")(norm9)
norm10 = keras.layers.BatchNormalization()(dense2)
dense3 = keras.layers.Dense(64, activation="swish")(norm10)
norm11 = keras.layers.BatchNormalization()(dense3)
dense4 = keras.layers.Dense(2, activation="sigmoid")(norm11)
model = keras.models.Model(inputs=input1, outputs=dense4)
adam = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=adam, loss="mse")

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, min_delta=1e-4)

model.fit(x=img, y=label, epochs=1000, batch_size=32, validation_split=0.1, callbacks=es)
model.save('Track_Model.h5')
