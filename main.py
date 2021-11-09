import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image

image = np.array(Image.open('data/input/arch.png').convert('L'))
n_sigmas = 5
for i in range(n_sigmas):
    edges = feature.canny(image, sigma=i * 2 + 1)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
