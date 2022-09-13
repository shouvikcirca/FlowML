from PIL import Image
import os
import numpy as np

a = os.listdir()[0]

im = Image.open('./'+a)
im = im.resize((128,128))
im = np.asarray(im)
im = np.expand_dims(im, 0)

np.save('testimage', im)
