import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# Load images from folders
def load_images_from_folder(folder_path):
    images = []
    file_types = ('*.png', '*.jpg', '*.jpeg')
    for file_type in file_types:
        for img_path in glob.glob(os.path.join(folder_path, file_type)):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img_gray)
    return images

# Canny edge detection
def edge_detection(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

# Hough line transform
def line_segments(image, rho, theta, threshold,minLineLength, maxLineGap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
    return lines


def line_orientation_histogram(lines, num_bins):
    orientations = []
    lengths = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            orientation = np.arctan2(dy, dx)

            # Convert orientation from radians to degrees
            orientation_degrees = np.degrees(orientation)

            orientations.append(orientation_degrees)
            lengths.append(length)

    # Update the range to (-180, 180) to match the degrees conversion
    histogram, _ = np.histogram(orientations, bins=num_bins, range=(-180, 180), weights=lengths)
    histogram = histogram / np.sum(histogram)

    return histogram



def circular_shift(arr, num_bins_shifted):
    return np.concatenate((arr[-num_bins_shifted:], arr[:-num_bins_shifted]))

def compute_angle_of_rotation(rotated_histograms, original_histograms, num_bins=10):
    match_results = []

    for rotated_hist in rotated_histograms:
        min_distance = float('inf')
        matched_original_idx = -1
        angle = 0
        
        for original_hist_idx, original_hist in enumerate(original_histograms):
            for shift in range(num_bins):
                shifted_rotated_hist = circular_shift(rotated_hist, shift)
                distance = np.linalg.norm(shifted_rotated_hist - original_hist)
                
                if distance < min_distance:
                    min_distance = distance
                    matched_original_idx = original_hist_idx
                    angle = shift * (360 / num_bins)
                    
        match_results.append((matched_original_idx, angle))
        
    return match_results

def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

#parameters that are used
canny_low_threshold = 100
canny_high_threshold = 150
hough_rho = 1
hough_theta = np.pi / 180
hough_threshold = 50
num_bins = 30
min_line_length = 100
max_line_gap = 10

original_books = load_images_from_folder('templateimages')
rotated_books = load_images_from_folder('rotatedimages')

# Canny Edge Detection
original_books_edges = [edge_detection(book, canny_low_threshold, canny_high_threshold) for book in original_books]
rotated_books_edges = [edge_detection(book, canny_low_threshold, canny_high_threshold) for book in rotated_books]

# output canny edge images
for i, edge_image in enumerate(original_books_edges):
    output_filename = f"original_book_edge{i}.png"
    cv2.imwrite(output_filename, edge_image)

for i, edge_image in enumerate(rotated_books_edges):
    output_filename = f"rotated_book_edge{i}.png"
    cv2.imwrite(output_filename, edge_image)



# Hough Line function
original_books_lines = [line_segments(edges, hough_rho, hough_theta, hough_threshold, min_line_length,max_line_gap) for edges in original_books_edges]
rotated_books_lines = [line_segments(edges, hough_rho, hough_theta, hough_threshold,min_line_length,max_line_gap) for edges in rotated_books_edges]

# draw lines inside of the images 
original_books_edges_with_lines = []
for img, lines in zip(original_books_edges, original_books_lines):
    img_with_lines = draw_lines(img.copy(), lines)
    original_books_edges_with_lines.append(img_with_lines)

rotated_books_edges_with_lines = []
for img, lines in zip(rotated_books_edges, rotated_books_lines):
    img_with_lines = draw_lines(img.copy(), lines)
    rotated_books_edges_with_lines.append(img_with_lines)


import os
# lines of the original outputs. Creates directory and puts images in it 
output_dir = 'Original Outputs'

# lines of the rotated outputs. Creates directory and puts images in it 
output_dirR = 'Rotated Outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dirR):
    os.makedirs(output_dirR)
for i, (original_edge_img, rotated_edge_img) in enumerate(zip(original_books_edges_with_lines, rotated_books_edges_with_lines)):
    cv2.imwrite(os.path.join(output_dir, f'Original_Book_Edges_{i+1}.png'), original_edge_img)
    cv2.imwrite(os.path.join(output_dirR, f'Rotated_Book_Edges_{i+1}.png'), rotated_edge_img)



# Line orientation part
original_books_histograms = [line_orientation_histogram(lines, num_bins) for lines in original_books_lines]
rotated_books_histograms = [line_orientation_histogram(lines, num_bins) for lines in rotated_books_lines]




# # display images with line segment, lines are red 
for idx, (edges, lines) in enumerate(zip(original_books_edges, original_books_lines)):
    img_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Original Book {} - Lines".format(idx+1), img_lines)


for idx, (edges, lines) in enumerate(zip(rotated_books_edges, rotated_books_lines)):
    img_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Rotated Book {} - Lines".format(idx+1), img_lines)

matches_and_angles = compute_angle_of_rotation(rotated_books_histograms, original_books_histograms)

# display matches between rotated images and original images
for i, (match_idx, angle) in enumerate(matches_and_angles):
    print(f'Rotated book {i + 1} matches with original book {match_idx + 1}, with a rotation angle of {angle:.2f} degrees')

# Display histograms
for idx, histogram in enumerate(original_books_histograms):
    plt.bar(range(num_bins), histogram)
    plt.title("Original Book {} - Histogram".format(idx+1))
    plt.show()

for idx, histogram in enumerate(rotated_books_histograms):
    plt.bar(range(num_bins), histogram)
    plt.title("Rotated Book {} - Histogram".format(idx+1))
    plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()
