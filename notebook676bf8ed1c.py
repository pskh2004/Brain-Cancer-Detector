# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'brain-cancer-detection-mri-images:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F5509309%2F9128964%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240827%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240827T135933Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D352465b4f1954d37cb1e0874d84ee9cc1f85de1bc995835ad98410916313aea67a71e0e0558b657c28d5a87277c9a4bdad6d1af227a6597ec08acbcee4584face6e24146c164940e8509253413da3678f6faf7b0f24103e8c470ebf00c4f3c8ae80f2d39d17fb17592ae97794e15788094d31edf367e1f38a1ebef005598521ba3d7c98814514aa939e15c21973d25802995397e7664bb4ffa70c07a84118a7f98395eaf1749c54543767fa646a774f5285dae4eecde5b6511973c8df9f040d37995890888543a6b082d44cfe1bf8e99c620ac6dd9a7cff105e253846309a38c045cf879276fd7be5edefcd5553f13bad452d6b4c8066729596bec0223985a9e'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

encoder=OneHotEncoder()
encoder.fit([[1],[0]])
#1=Tumor
#0=Normal

paths = []
result= []
data=[]
for r, d, f in os.walk(r'../input/brain-cancer-detection-mri-images/Dataset/healthy'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))
for path in paths :
  img=Image.open(path)
  img=img.resize((128,128))
  img=np.array(img)
  if(img.shape==(128,128,3)):
    data.append(np.array(img))
    result.append(encoder.transform([[0]]).toarray())

paths = []
for r, d, f in os.walk(r'../input/brain-cancer-detection-mri-images/Dataset/tumor'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))
for path in paths :
  img=Image.open(path)
  img=img.resize((128,128))
  img=np.array(img)
  if(img.shape==(128,128,3)):
    data.append(np.array(img))
    result.append(encoder.transform([[1]]).toarray())
print(data)

data = np.array(data)
data.shape

result = np.array(result)
result= result.reshape(607,2)

x_train,x_test,y_train,y_test=train_test_split(data,result,test_size=0.2,shuffle=True,random_state=0)

from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(128, 128, 3), padding='Same', kernel_regularizer=l2(0.001)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same', kernel_regularizer=l2(0.001)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='Adamax')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

print(model.summary())

y_train.shape

history = model.fit(x_train, y_train, epochs=30, batch_size=40, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping, reduce_lr])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()

def names(number):
    if number==0:
        return 'No, Its not a tumor'
    else:
        return 'Its a Tumor'
from sklearn.metrics import accuracy_score
m=model.predict(x_test)
m=np.argmax(m,axis=1)
y_testt=np.argmax(y_test,axis=1)
ac=accuracy_score(y_testt,m)
print(ac)

img = Image.open(r"../input/brain-cancer-detection-mri-images/Dataset/10 no.jpg")
x = np.array(img.resize((128, 128)))
x = x.reshape(1, 128, 128, 3)

# Predict the class
res = model.predict_on_batch(x)
classification = np.argmax(res)

# Display the image and classification result
plt.imshow(img)
print("This Is", names(classification))
