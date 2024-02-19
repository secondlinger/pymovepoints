import math
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np

from ImageResizer import ImageResizer
from fileselector import VideoFileSelector

# Initialize lists to store coordinates
points_list_img1 = []
points_list_img2 = []


points_list_10met  = []

a = 0.1
b = -0.2
c = 0.001
d = 0.001
e = 0.0

class TrackbarCallbacks:
    def __init__(self):
        pass

    @staticmethod
    def on_fov_change_x(trackbarValue):
        global fovx
        fovx = trackbarValue

    @staticmethod
    def on_fov_change_y(trackbarValue):
        global fovy
        fovy = trackbarValue

    @staticmethod
    def on_A(trackbarValue):
        global a
        a = trackbarValue

    @staticmethod
    def on_B(trackbarValue):
        global b
        b = trackbarValue

    @staticmethod
    def on_C(trackbarValue):
        global c
        c = trackbarValue

    @staticmethod
    def on_D(trackbarValue):
        global d
        d = trackbarValue

    @staticmethod
    def on_E(trackbarValue):
        global e
        e = trackbarValue

# This function will remain mostly unchanged

class ImageFunctions:
    def __init__(self):
        pass

    @staticmethod
    def draw_points(img, points_list, frame_name):
        for point in points_list:
            cv2.circle(img, point, 5, (255, 0, 0), -1)
        cv2.imshow(frame_name, img)

    @staticmethod
    def click_event_img1(event, x, y, flags, params):
        global frame_img1
        handle_click_event(event, x, y, points_list_img1, frame_img1, 'Image1')

    @staticmethod
    def click_event_img2(event, x, y, flags, params):
        global img2
        handle_click_event(event, x, y, points_list_img2, img2, 'Image2')

    @staticmethod
    def click_event_img3(event, x, y, flags, params):
        global OriginalImage
        handle_click_event10met(event, x, y, points_list_10met, OriginalImage, 'Image2')

    @staticmethod
    def draw_on(points_list, img):
        for point in points_list:
            cv2.circle(img, point, 5, (255, 0, 0), -1)
        return img



# No changes here
def handle_click_event(event, x, y, points_list, img, window_name):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_list) < 4:
            points_list.append((x, y))
            ImageFunctions.draw_points(img, points_list,window_name)
            print(f"Point added to {window_name}: ({x},{y})")
        else:
            print(f"Maximum points reached for {window_name}.")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points_list:
            points_list.pop()
            img_copy = frame.copy() if window_name == 'Image1' else original_img2.copy()
            ImageFunctions.draw_points(img_copy, points_list,window_name)
            if window_name == 'Image1':
                global frame_img1
                frame_img1 = img_copy
            else:
                global img2
                img2 = img_copy
            print(f"Last point removed from {window_name}.")

def handle_click_event10met(event, x, y, points_list, img, window_name):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_list) < 20:
            points_list.append((x, y))
            ImageFunctions.draw_points(img, points_list,window_name)
            print(f"Point added to {window_name}: ({x},{y})")
        else:
            print(f"Maximum points reached for {window_name}.")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points_list:
            points_list.pop()
            img_copy = img.copy()
            ImageFunctions.draw_points(img_copy, points_list,window_name)
            if window_name == 'Original Image with Source Points':
                global frame_img1
                frame_img1 = img_copy
            else:
                global img2
                img2 = img_copy
            print(f"Last point removed from {window_name}.")


def blend_images(transformed_img, img2, alpha=0.5):
    if transformed_img is not None:
        # Ensure the dimensions match, resizing img2 if necessary
        if transformed_img.shape[:2] != img2.shape[:2]:
            img2_resized = cv2.resize(img2, (transformed_img.shape[1], transformed_img.shape[0]))
        else:
            img2_resized = img2

        beta = 1.0 - alpha  # Calculate the weight for the second image
        gamma = 0  # Scalar added to each sum, usually zero

        # Blend the images together
        blended_image = cv2.addWeighted(transformed_img, alpha, img2_resized, beta, gamma)

        return blended_image


def fov_to_intrinsic(fov_x, fov_y, width, height):
    # Convert FoV from degrees to radians
    fov_x_rad = math.radians(fov_x)
    fov_y_rad = math.radians(fov_y)

    # Calculate focal lengths
    fx = width / (2 * math.tan(fov_x_rad / 2))
    fy = height / (2 * math.tan(fov_y_rad / 2))

    # Calculate principal point (assuming it's at the center of the image)
    cx = width / 2
    cy = height / 2

    # Construct the intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K

def adjust_fov(frame, fov_x, fov_y, img_size):
    # Modify here with actual distortion coefficients from calibration
    dist_coeffs = np.array([a/100000,-b/100000,c/10000/100,d/10000/100,-e/10000])  # Example distortion coefficients

    # Assuming the typo in 'img2.shape' is corrected to 'frame.shape'
    camera_matrix = fov_to_intrinsic(fov_x, fov_y, frame.shape[0], frame.shape[1])

    # Apply the transformation
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, img_size, 1, img_size)
    adjusted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image
    x, y, w, h = roi
    adjusted_frame = adjusted_frame[y:y + h, x:x + w]
    return adjusted_frame


def apply_perspective_transform_and_map_points(original_img,src_points , matrix):
        # Transform src_points from target_img to original_img
        transformed_points = []
        for pt in src_points:
            point_homogenous = np.array([*pt, 1]).reshape(-1, 1)
            transformed_point = matrix.dot(point_homogenous)
            transformed_point = transformed_point / transformed_point[2]  # Normalize to convert from homogenous coordinates
            transformed_point = transformed_point[:2].astype(int).reshape(-1)  # Convert coordinates to integer
            transformed_points.append((transformed_point[0], transformed_point[1]))
            # Draw the transformed points on the transformed image
            cv2.circle(transformed_img, (transformed_point[0], transformed_point[1]), 5, (0, 255, 0), -1)

        # Draw the src_points on the original image for reference
        for pt in src_points:
            cv2.circle(original_img, pt, 30, (0, 0, 255), -1)

        return transformed_img, transformed_points

def apply_perspective_transform():
    if len(points_list_img1) == 4 and len(points_list_img2) == 4:
        pts1 = np.float32(points_list_img1)
        pts2 = np.float32(points_list_img2)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_img = cv2.warpPerspective(original_frame_img1, matrix, (img2.shape[1], img2.shape[0]))
        cv2.imshow('Transformed Image', transformed_img)
        return transformed_img, matrix
    else:
        return None, None


video_selector = VideoFileSelector()

# Now call the function to get the paths
video_path, img2_path = video_selector.select_video_file()

# Load the second image
original_img2 = cv2.imread(img2_path)
OriginalImage = cv2.imread(img2_path)
img2 = original_img2.copy()

# Initialize video capture
cap = cv2.VideoCapture(video_path)

#----------------------------------------------------------------------------------
# Windows for display
cv2.namedWindow('Image')
cv2.namedWindow('Image1')
cv2.namedWindow('Image2')
cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original Image with Source Points', cv2.WINDOW_NORMAL)


cv2.setMouseCallback('Image1', ImageFunctions.click_event_img1)
cv2.setMouseCallback('Image2', ImageFunctions.click_event_img2)
cv2.setMouseCallback('Original Image with Source Points', ImageFunctions.click_event_img3)


# Create trackbars for FOV adjustment
cv2.createTrackbar('FOV X', 'Image', 10, 1000, TrackbarCallbacks.on_fov_change_x)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('FOV Y', 'Image', 10, 1000, TrackbarCallbacks.on_fov_change_y)  # Adjust the range as needed

# Create trackbars for FOV adjustment
cv2.createTrackbar('A', 'Image', 0, 10000, TrackbarCallbacks.on_A)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('B', 'Image', 0, 10000, TrackbarCallbacks.on_B)  # Adjust the range as needed
cv2.createTrackbar('C', 'Image', 0, 1000, TrackbarCallbacks.on_C)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('D', 'Image', 0, 1000, TrackbarCallbacks.on_D)  # Adjust the range as needed
cv2.createTrackbar('E', 'Image', 0, 1000, TrackbarCallbacks.on_E)  # Assuming FOV range from 0 to 180

#----------------------------------------------------------------------------------




scale = 2

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
        continue

    if  fovx==0:
        key = cv2.waitKey(1) & 0xFF
        continue

    if  fovy==0:
        key = cv2.waitKey(1) & 0xFF
        continue

    img2 = original_img2

    # Resize the frame to match img2's display size if needed
    frame_img1 = cv2.resize(frame, (frame.shape[1]*scale, frame.shape[0]*scale))
    img2 = cv2.resize(img2, (img2.shape[1]*scale, img2.shape[0]*scale))

   # frame = ResizeWithAspectRatioAndFill(frame_img1, width=1780, height=1000)  # Resize by width OR
    img2 = ImageResizer.ResizeWithAspectRatioAndFill(img2,width=1780, height=1000)  # Resize by width OR

    k = 1.0
    kernel = np.array([[-k, -k, -k], [-k, 1 + 8 * k, -k], [-k, -k, -k]])
    frame_img1 = cv2.filter2D(frame_img1, ddepth=-1, kernel=kernel)
    frame_img2 = cv2.filter2D(img2, ddepth=-1, kernel=kernel)

    frame_img1 = adjust_fov(frame_img1, fovx/10, fovy/10, (frame_img1.shape[1], frame_img1.shape[0]))
    original_frame_img1 = frame_img1.copy()

    if len(points_list_img1) > 0:
        ImageFunctions.draw_points(frame_img1, points_list_img1, 'Image1')
    if len(points_list_img2) > 0:
        ImageFunctions.draw_points(frame_img2, points_list_img2, 'Image2')

    cv2.imshow('Image1', frame_img1)
    cv2.imshow('Image2', frame_img2)

    transformed_img, matrix = apply_perspective_transform()

    if matrix is not None:
        transformed_image, transformed_points = apply_perspective_transform_and_map_points(original_frame_img1, points_list_10met, matrix)

    if transformed_img is not None:
        original_img_with_points = ImageFunctions.DrawOn(points_list_10met, original_frame_img1)

        cv2.imshow('Original Image with Source Points', original_img_with_points)
        cv2.imshow('Transformed Image with Transformed Points', transformed_image)

        blended_image = blend_images(transformed_img, img2, 0.5)

        if transformed_img is not None:
            cv2.imshow('Image3', blended_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
