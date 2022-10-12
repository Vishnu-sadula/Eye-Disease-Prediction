import numpy as np
import pandas as pd
import cv2
import random
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("input/ocular-disease-recognition-odir5k/full_df.csv")
df.head()


def has_cataract(text):
    if "cataract" in text:
        return 1
    else:
        return 0


df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
df["right_cataract"][1:5]
left_cataract = df.loc[(df.C == 1) & (df.left_cataract == 1)]["Left-Fundus"].values
right_cataract = df.loc[(df.C == 1) & (df.right_cataract == 1)]["Right-Fundus"].values
right_cataract[:15]
print("Number of images in left cataract: {}".format(len(left_cataract)))
print("Number of images in right cataract: {}".format(len(right_cataract)))
left_normal = df.loc[(df.C == 0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250,
                                                                                                              random_state=42).values
right_normal = df.loc[(df.C == 0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250,
                                                                                                                 random_state=42).values
right_normal[:15]
cataract = np.concatenate((left_cataract, right_cataract), axis=0)
normal = np.concatenate((left_normal, right_normal), axis=0)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

dataset_dir = "input/ocular-disease-recognition-odir5k/preprocessed_images"
image_size = 224
labels = []
dataset = []


def create_dataset(image_category, label):
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir, img)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (image_size, image_size))

        except:
            continue

        dataset.append([np.array(image), np.array(label)])
    random.shuffle(dataset)
    return dataset


dataset = create_dataset(cataract, 1)
dataset = create_dataset(normal, 0)
len(dataset)
plt.figure(figsize=(12, 7))
for i in range(10):
    sample = random.choice(range(len(dataset)))
    image = dataset[sample][0]
    category = dataset[sample][1]
    if category == 0:
        label = "Normal"
    else:
        label = "Cataract"
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.xlabel(label)
plt.tight_layout()
x = np.array([i[0] for i in dataset]).reshape(-1, image_size, image_size, 3)
y = np.array([i[1] for i in dataset])
x.shape
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from tensorflow.keras.applications.vgg19 import VGG19

vgg = VGG19(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("vgg19.h5", monitor="val_acc", verbose=1, save_best_only=True,
                             save_weights_only=False, period=1)
earlystop = EarlyStopping(monitor="val_acc", patience=5, verbose=1)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test),
                    verbose=1, callbacks=[checkpoint, earlystop])
print(history.history.keys())
import matplotlib.pyplot as plt
% matplotlib
inline
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
loss, accuracy = model.evaluate(x_test, y_test)
print("loss:", loss)
print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

y_pred = model.predict_classes(x_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
!pip
install
mlxtend
from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=cm, figsize=(8, 7), class_names=["Normal", "Cataract"],
                      show_normed=True);
plt.figure(figsize=(12, 7))
for i in range(10):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]

    if category == 0:
        label = "Normal"
    else:
        label = "Cataract"

    if pred_category == 0:
        pred_label = "Normal"
    else:
        pred_label = "Cataract"

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label, pred_label))
plt.tight_layout()