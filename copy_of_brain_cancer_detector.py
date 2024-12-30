
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
for r, d, f in os.walk(r'/content/Dataset/healthy'):
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
for r, d, f in os.walk(r'/content/Dataset/tumor'):
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

from sklearn.metrics import roc_curve, auc

# محاسبه ROC Curve
fpr, tpr, thresholds = roc_curve(y_testt, m)
roc_auc = auc(fpr, tpr)

# رسم نمودار ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve

# محاسبه Precision و Recall
precision, recall, _ = precision_recall_curve(y_testt, m)

# رسم نمودار Precision-Recall
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

from sklearn.metrics import classification_report

# چاپ گزارش طبقه‌بندی
print(classification_report(y_testt, m, target_names=['No Tumor', 'Tumor']))

# رسم توزیع پیش‌بینی‌ها
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.histplot(m, bins=2, kde=False, color='blue')
plt.xticks([0, 1], ['No Tumor', 'Tumor'])
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.title('Distribution of Predictions')
plt.show()

img = Image.open(r"../input/brain-cancer-detection-mri-images/Dataset/10 no.jpg")
x = np.array(img.resize((128, 128)))
x = x.reshape(1, 128, 128, 3)

# Predict the class
res = model.predict_on_batch(x)
classification = np.argmax(res)

# Display the image and classification result
plt.imshow(img)
print("This Is", names(classification))

# Mount Google Drive (optional, if the ZIP file is in your Drive)
"""from google.colab import drive
drive.mount('/content/drive')"""

# Unzip the file
!unzip /content/archive.zip

!unzip /content/archive.zip # If the file is in the main Colab directory
