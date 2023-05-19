from tensorflow.keras.models import load_model
import tensorflow
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model('model_data/best_model_lr0.001.h5')
print("Successfully load the model!")

# load image
img1 = image.load_img('./data/test/cats/cat.4495.jpg', target_size=(150, 150))
img = image.img_to_array(img1)
img = img / 255

img = np.expand_dims(img, axis=0)  # create a batch of size 1 [N,H,W,C]
prediction = model.predict(img, batch_size=None, steps=1)  # gives all class prob.

if prediction[:, :] > 0.5:
    value = 'Dog :%1.2f' % (prediction[0, 0])
    plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
else:
    value = 'Cat :%1.2f' % (1.0 - prediction[0, 0])
    plt.text(20, 62, value, color='red', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))

plt.imshow(img1)
plt.savefig("figures/lr0.001/a.png",bbox_inches='tight')
plt.show()
