import skimage.io
import skimage.color
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import HOG

def compare_histograms(hist1, hist2):
    similarity = 0
    threshold = 0.8
    
    x_2d = np.vstack(hist1)
    
    similarity_matrix = cosine_similarity(x_2d, hist2.reshape(1, -1))
    
    similarity_percentage = np.mean(similarity_matrix) * 100
    
    print("Similarity percentage : " + str(similarity_percentage))
    
    if similarity >= threshold:
        print("This images contains a ball.")
    else:
        print("This Image is not contain a ball.")

ball_dataset_histograms = []


i = 1
while i <= 51:
    
    print("Training Image " + str(i) + "...")
    
    if i > 50:
        img = skimage.io.imread("c:\Coding\Python\Image Processing\ImageRecognition\datatest\datatest1.jpg")
    elif i >= 10:
        img = skimage.io.imread("c:\Coding\Python\Image Processing\ImageRecognition\dataset\\ball\\0" + str(i) + ".jpg")
    else:
        img = skimage.io.imread("c:\Coding\Python\Image Processing\ImageRecognition\dataset\\ball\\00" + str(i) + ".jpg")

    img_gray = skimage.color.rgb2gray(img)

    horizontal_mask = np.array([-1, 0, 1])
    vertical_mask = np.array([[-1],
                              [0],
                              [1]])

    horizontal_gradient = HOG.calculate_gradient(img_gray, horizontal_mask)
    vertical_gradient = HOG.calculate_gradient(img_gray, vertical_mask)

    grad_magnitude = HOG.gradient_magnitude(horizontal_gradient, vertical_gradient)
    grad_direction = HOG.gradient_direction(horizontal_gradient, vertical_gradient)

    grad_direction = grad_direction % 180
    hist_bins = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    # Histogram of the first cell in the first block.
    cell_direction = grad_direction[:8, :8]
    cell_magnitude = grad_magnitude[:8, :8]
    HOG_cell_hist = HOG.HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)

    if i<=50:
        ball_dataset_histograms.append(HOG_cell_hist)
    else:
        hist1 = HOG_cell_hist
    # Ball detection logic
    i+=1

is_ball = compare_histograms(ball_dataset_histograms, hist1)