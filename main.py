import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image
import matplotlib.lines as mlines


def find_nearest(point, points):
    dist = np.sum((points - point) ** 2, axis=1)
    nearest = points[np.argmin(dist)]
    return nearest


def find_far_points(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return (x1 - 1000, a * (x1 - 1000) + b), (x2 + 1000, a * (x2 + 1000) + b)


def get_two_lines_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        return float('inf'), float('inf')
    return x / z, y / z


def filter_all_lines(line_points, image_height, image_width):
    for l1 in range(line_points.shape[0]):
        intersections = []
        for l2 in range(line_points.shape[0]):
            if l2 == l1:
                continue
            x, y = get_two_lines_intersect((line_points[l1, 0], line_points[l1, 1]),
                                           (line_points[l1, 2], line_points[l1, 3]),
                                           (line_points[l2, 0], line_points[l2, 1]),
                                           (line_points[l2, 2], line_points[l2, 3]))
            if (0 <= x <= image_height) and (0 <= y <= image_width):
                intersections.append((x, y))
        x1 = line_points[l1, 0]
        y1 = line_points[l1, 1]
        x2 = line_points[l1, 2]
        y2 = line_points[l1, 3]
        neg, pos = find_far_points((x1, y1), (x2, y2))
        if len(intersections) >= 2:

            intersections_np = np.array(intersections)
            for_sign = intersections_np[:, 0] - x1
            _for_sign = x2 - x1
            that = np.argwhere(np.sign(for_sign) == np.sign(_for_sign)).squeeze(1)
            this = np.argwhere(np.sign(for_sign) != np.sign(_for_sign)).squeeze(1)

            if that.size != 0:
                nearest_1 = find_nearest(np.array([x2, y2]), intersections_np[that])
            else:
                if x2 - x1 > 0:
                    nearest_1 = pos
                else:
                    nearest_1 = neg
            if this.size != 0:
                nearest_2 = find_nearest(np.array([x1, y1]), intersections_np[this])
            else:
                if x2 - x1 > 0:
                    nearest_2 = neg
                else:
                    nearest_2 = pos
            line_points[l1, 0] = nearest_1[0]
            line_points[l1, 1] = nearest_1[1]
            line_points[l1, 2] = nearest_2[0]
            line_points[l1, 3] = nearest_2[1]
        else:
            line_points[l1, 0] = neg[0]
            line_points[l1, 1] = neg[1]
            line_points[l1, 2] = pos[0]
            line_points[l1, 3] = pos[1]
    return line_points


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
    line_points = np.empty(shape=(len(line_rhos), 4))
    for i, (rho, theta) in enumerate(zip(line_rhos, line_thetas)):
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + edge_width_half
        y0 = (b * rho) + edge_height_half
        line_points[i, 0] = x0 - 10 * b
        line_points[i, 1] = y0 + 10 * a
        line_points[i, 2] = x0 + 10 * b
        line_points[i, 3] = y0 - 10 * a
    # line_points = filter_all_lines(line_points, image.shape[0], image.shape[1])
    for row in line_points:
        subplot4.add_line(mlines.Line2D([row[0], row[2]], [row[1], row[3]]))
    subplot3.invert_yaxis()
    subplot3.invert_xaxis()

    subplot1.title.set_text("Original Image")
    subplot2.title.set_text("Edge Image")
    subplot3.title.set_text("Hough Space")
    subplot4.title.set_text("Detected Lines")
    plt.show()
    return accumulator.T, rhos, thetas


img = np.array(Image.open('data/input/arch.png').convert('L'))
edges = feature.canny(img, sigma=3)

line_detection_vectorized(img, edges, threshold=200)
