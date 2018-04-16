from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras import optimizers

# load data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# the number of images that will be processed in a single step
batch_size=32
# the size of the images that we'll learn on - we'll use their natural size
image_size=(240, 800)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = test_datagen.flow_from_directory(
        'data/valid',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

num_classes = len(train_generator.class_indices)

def hack_resnet(input_size, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_size)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)
    
    # this is the model we will train
    newmodel = Model(inputs=base_model.input, outputs=x)

    return newmodel

model = hack_resnet(train_generator.image_shape, num_classes)

# set weights in all but last layer
# to non-trainable (weights will not be updated)
for layer in model.layers[:len(model.layers)-2]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# Fit the model
model.fit_generator(
        train_generator,
        steps_per_epoch=3474 // batch_size, 
        validation_data=valid_generator,
        validation_steps=395 // batch_size,
        epochs=10,
        verbose=1)

# Final evaluation of the model
test_steps_per_epoch = numpy.math.ceil(float(test_generator.samples) / test_generator.batch_size)
raw_predictions = model.predict_generator(test_generator, steps=test_steps_per_epoch)
predictions = numpy.argmax(raw_predictions, axis=1)

print("Prediction Distribution:  " + str(numpy.bincount(predictions)))
print("Groundtruth Distribution: " + str(numpy.bincount(test_generator.classes)))

from sklearn import metrics
class_labels = [item[0] for item in sorted(test_generator.class_indices.items(), key=lambda x: x[1])] #get a list of classes
print(metrics.classification_report(test_generator.classes, predictions, target_names=class_labels))
