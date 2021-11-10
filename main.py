import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image
import matplotlib.lines as mlines


def line_detection_vectorized(image, edge_image, num_rhos=180, num_thetas=180, threshold=220):
    edge_height_half, edge_width_half = image.shape[0] / 2, image.shape[1] / 2
    diag = np.sqrt(np.square(image.shape[0]) + np.square(image.shape[1]))
    delta_theta = 180 / num_thetas
    delta_rho = (2 * diag) / num_rhos
    #
    thetas = np.arange(0, 180, step=delta_theta)
    #
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    figure = plt.figure(figsize=(12, 12))
    subplot1 = figure.add_subplot(1, 4, 1)
    subplot1.imshow(image, cmap="gray")
    subplot2 = figure.add_subplot(1, 4, 2)
    subplot2.imshow(edge_image, cmap="gray")
    subplot3 = figure.add_subplot(1, 4, 3)
    subplot3.set_facecolor((0, 0, 0))
    subplot4 = figure.add_subplot(1, 4, 4)
    subplot4.imshow(image, cmap='gray')
    # center of image is 0,0
    edge_points = np.argwhere(edge_image != 0) - np.array([[edge_height_half, edge_width_half]])
    #
    # for each point we have:
    # rho = x * cos(theta) + y * sin(theta)
    image_rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas]))
    #
    for ys in image_rho_values:
        subplot3.plot(thetas, ys, color="white", alpha=0.05)

    rhos = np.arange(-diag, diag, step=delta_rho)

    accumulator, _, __ = np.histogram2d(np.tile(thetas, edge_points.shape[0]).ravel(),
                                        image_rho_values.ravel(),
                                        bins=[thetas, rhos])
    lines = np.argwhere(accumulator.T > threshold)
    rho_ids, theta_ids = lines[:, 0], lines[:, 1]
    line_rhos, line_thetas = rhos[rho_ids], thetas[theta_ids]
    subplot3.plot([line_thetas], [line_rhos], color="yellow", marker='o')

    for (rho, theta) in zip(line_rhos, line_thetas):
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

    subplot3.invert_yaxis()
    subplot3.invert_xaxis()

    subplot1.title.set_text("Original Image")
    subplot2.title.set_text("Edge Image")
    subplot3.title.set_text("Hough Space")
    subplot4.title.set_text("Detected Lines")
    plt.show()
    return accumulator.T, rhos, thetas


image = np.array(Image.open('data/input/arch.png').convert('L'))
edges = feature.canny(image, sigma=3)

line_detection_vectorized(image, edges)
