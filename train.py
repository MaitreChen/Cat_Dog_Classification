# IMPORT PACKAGE
from prettytable import PrettyTable
import datetime
import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Create model dir
MODEL_PATH = './model_data'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Define image attribute
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CLASSES = 2

# Define hyper-parameters
BATCH_SIZE = 64
EPOCHS = 40
lr = 0.001
optimizer = Adam(learning_rate=lr)

# Add to table
# Create the table object, name, and alignment
table = PrettyTable(['Hyper-Parameters & data infos', 'Value'])
table.align['Hyper-Parameters & data infos'] = 'l'
table.align['Value'] = 'r'

table.add_row(['BATCH_SIZE', BATCH_SIZE])
table.add_row(['EPOCHS', EPOCHS])
table.add_row(['LR', lr])
table.add_row(['optimizer', optimizer])
table.add_row(['', ''])

table.add_row(['IMG_HEIGHT', IMG_HEIGHT])
table.add_row(['IMG_WIDTH ', IMG_WIDTH])
table.add_row(['NUM_CLASSES ', NUM_CLASSES])
print(table)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# Load dataset
train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True)
val_data_gen = ImageDataGenerator(rescale=1. / 255)

# Training Set
train_set = train_data_gen.flow_from_directory('./data/train',
                                               target_size=(
                                                   IMG_WIDTH, IMG_HEIGHT),
                                               batch_size=BATCH_SIZE,
                                               class_mode='binary')
# Validation Set
val_set = val_data_gen.flow_from_directory('./data/val',
                                           target_size=(IMG_WIDTH, IMG_HEIGHT),
                                           batch_size=BATCH_SIZE,
                                           class_mode='binary',
                                           shuffle=False)
# Test Set
test_set = val_data_gen.flow_from_directory('./data/test',
                                            target_size=(
                                                IMG_WIDTH, IMG_HEIGHT),
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary',
                                            shuffle=False)

epoch_steps = len(train_set)
val_epoch_steps = len(val_set)

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
best_model = ModelCheckpoint(f'{MODEL_PATH}/best_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min')

# Train model with callbacks
history = model.fit(
    train_set,
    steps_per_epoch=epoch_steps,
    epochs=EPOCHS,
    validation_data=val_set,
    validation_steps=val_epoch_steps,
    callbacks=[tensorboard_callback, early_stop, best_model]
)

# Evaluate model
train_info = model.evaluate(train_set)
val_info = model.evaluate(val_set)
test_info = model.evaluate(test_set)

print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f' %
      (train_info[1] * 100, train_info[0]))
print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f' %
      (val_info[1] * 100, val_info[0]))
print('Testing Accuracy: %1.2f%%     Testing loss: %1.6f' %
      (test_info[1] * 100, test_info[0]))
