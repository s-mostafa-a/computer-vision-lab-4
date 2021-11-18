import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image
import matplotlib.lines as mlines


def get_min_max_x_y(indices):
    min_x = np.min(indices[:, 0])
    max_x = np.max(indices[:, 0])

    min_y = np.min(indices[:, 1])
    max_y = np.max(indices[:, 1])

    return min_x, max_x, min_y, max_y


def get_2d_indices(one_d_indices, shape):
    x = shape[0]
    y = shape[1]
    x_i = (np.floor(one_d_indices / y)).astype(int)
    y_i = one_d_indices % y
    assert (x_i < x).all()
    return np.column_stack((x_i, y_i))


def find_voters(edges_theta_values, edges_rho_values, range_thetas, range_rhos, vote_thetas, vote_rhos, shape):
    range_theta_distance = range_thetas[1] - range_thetas[0]
    range_rhos_distance = range_rhos[1] - range_rhos[0]
    group_of_indices_per_edge = []
    for vote_theta, vote_rho in zip(vote_thetas, vote_rhos):
        indices_of_thetas_in_condition = np.argwhere(
            (vote_theta <= edges_theta_values) & (edges_theta_values < vote_theta + range_theta_distance)).squeeze(1)
        indices_of_rhos_in_condition = np.argwhere(
            (vote_rho <= edges_rho_values) & (edges_rho_values < vote_rho + range_rhos_distance)).squeeze(1)
        indices = get_2d_indices(
            np.array(list(set(indices_of_thetas_in_condition) & set(indices_of_rhos_in_condition))), shape=shape)
        group_of_indices_per_edge.append(indices)
    return group_of_indices_per_edge


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
    edge_points_x_y = np.argwhere(edge_image != 0) - np.array([[edge_height_half, edge_width_half]])
    #
    # for each point we have:
    # rho = x * cos(theta) + y * sin(theta)
    edges_rho_values = np.matmul(edge_points_x_y, np.array([sin_thetas, cos_thetas]))
    #
    for ys in edges_rho_values:
        subplot3.plot(thetas, ys, color="white", alpha=0.05)

    rhos = np.arange(-diag, diag, step=delta_rho)
    accumulator, _, __ = np.histogram2d(np.tile(thetas, edge_points_x_y.shape[0]),
                                        edges_rho_values.ravel(),
                                        bins=[thetas, rhos])
    lines = np.argwhere(accumulator.T > threshold)
    rho_ids, theta_ids = lines[:, 0], lines[:, 1]
    line_rhos, line_thetas = rhos[rho_ids], thetas[theta_ids]
    group_of_indices_per_edge = find_voters(edges_theta_values=np.tile(thetas, edge_points_x_y.shape[0]),
                                            edges_rho_values=edges_rho_values.ravel(), range_thetas=thetas,
                                            range_rhos=rhos, vote_thetas=line_thetas, vote_rhos=line_rhos,
                                            shape=edges_rho_values.shape)
    line_points = []
    for indices in group_of_indices_per_edge:
        xys = edge_points_x_y[indices[:, 0]]
        min_x, max_x, min_y, max_y = get_min_max_x_y(xys)
        line_points.append(
            [min_x + edge_height_half, min_y + edge_width_half, max_x + edge_height_half, max_y + edge_width_half])
    subplot3.plot([line_thetas], [line_rhos], color="yellow", marker='o')
    # line_points = np.empty(shape=(len(line_rhos), 4))
    # for i, (rho, theta) in enumerate(zip(line_rhos, line_thetas)):
    #     a = np.cos(np.deg2rad(theta))
    #     b = np.sin(np.deg2rad(theta))
    #     x0 = (a * rho) + edge_width_half
    #     y0 = (b * rho) + edge_height_half
    #     line_points[i, 0] = x0 - 10 * b
    #     line_points[i, 1] = y0 + 10 * a
    #     line_points[i, 2] = x0 + 10 * b
    #     line_points[i, 3] = y0 - 10 * a
    # line_points = filter_all_lines(line_points, image.shape[0], image.shape[1])
    for row in line_points:
        subplot4.add_line(mlines.Line2D([row[1], row[3]], [row[0], row[2]]))
    subplot3.invert_yaxis()
    subplot3.invert_xaxis()

    subplot1.title.set_text("Original Image")
    subplot2.title.set_text("Edge Image")
    subplot3.title.set_text("Hough Space")
    subplot4.title.set_text("Detected Lines")
    plt.show()
    return accumulator.T, rhos, thetas


if __name__ == '__main__':
    img = np.array(Image.open('data/input/arch.png').convert('L'))
    edges = feature.canny(img, sigma=3)

    line_detection_vectorized(img, edges, threshold=220)
