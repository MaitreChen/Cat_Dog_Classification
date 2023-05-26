from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def predict(img_path, model_path, save_dir):
    # load model
    model = load_model(model_path)
    print("Successfully load the model!")

    # load image and pre-process
    img1 = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img1)
    img = img / 255

    img = np.expand_dims(img, axis=0)  # create a batch of size 1 [N,H,W,C]
    prediction = model.predict(img, batch_size=None, steps=1)  # gives all class prob.

    # predict
    if prediction[:, :] > 0.5:
        value = 'Dog :%1.2f' % (prediction[0, 0])
        plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
    else:
        value = 'Cat :%1.2f' % (1.0 - prediction[0, 0])
        plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))

    # plot the result
    img = np.squeeze(img, 0)
    img_file = img_path.split('/')[-1]
    save_path = os.path.join(save_dir, img_file).replace('\\', '/')
    plt.imshow(img)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    SAVE_DIR = './figures'
    MODEL_DIR = './model_data'
    IMAGE_PATH = './img/img1.jpg'
    MODEL_PATH = f'{MODEL_DIR}/best_model_lr0.001.h5'

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    predict(IMAGE_PATH, MODEL_PATH, SAVE_DIR)
