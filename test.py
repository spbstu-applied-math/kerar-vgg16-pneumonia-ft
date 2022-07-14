import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
import cv2

model = keras.models.load_model("MyModel.h5")



image_size = 224

test_dir = Path("test")

def prepare_and_load(dir):

    normal_dir=dir/'NORMAL'
    pneumonia_dir=dir/'PNEUMONIA'

    normal_cases = normal_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_dir.glob('*.jpeg')
    data,labels=([] for x in range(2))
    def prepare(case):
        for img in case:
            img = cv2.imread(str(img))
            img = cv2.resize(img, (224,224))
            if img.shape[2] ==1:
                 img = np.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.
            if case==normal_cases:
                label = 0#keras.utils.to_categorical(0, num_classes=2)
            else:
                label = 1#keras.utils.to_categorical(1, num_classes=2)
            data.append(img)
            labels.append(label)
        return data,labels
    prepare(normal_cases)
    d,l=prepare(pneumonia_cases)
    d=np.array(d)
    l=np.array(l)
    return d,l


test_d, test_l = prepare_and_load(test_dir)

result = model.evaluate(test_d, test_l, 16)


json.dump(
    obj = {
        'accuracy': result[1]
    },
    fp = open('eval.json', 'w')
)