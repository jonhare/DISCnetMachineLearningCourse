from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
import numpy as np

model = ResNet50(include_top=False,
            weights='imagenet',
            pooling='avg')

img_path = 'data/test/Alilaguna/20130412_064059_20202.jpg'
img = image.load_img(img_path, target_size=(240, 800))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)

print(features.shape)
print(features)