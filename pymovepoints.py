import math
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

from pyonchangeGuiclass import ImageClickHandler
from pyonchangeclass import TrackbarHandler

# Initialize lists to store coordinates
points_list_img1 = []
points_list_img2 = []

handler = TrackbarHandler()
click_handler = ImageClickHandler()


def select_video_file():
    # Create a Tkinter root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select a video file
    video_path = filedialog.askopenfilename(title='Select Video File', filetypes=[('Video files', '*.mp4;*.avi;*.mov')])

    if video_path:  # If a file was selected
        # Extract the directory of the selected video
        video_directory = os.path.dirname(video_path)
        # Extract the base name of the video file (without extension)
        base_name = os.path.basename(video_path).rsplit('.', 1)[0]
        # Construct the img2 path to be in the "map" directory within the same directory as the video file
        img2_directory = os.path.join(video_directory, "map")
        img2_path = os.path.join(img2_directory, f"{base_name}.png")

        # Check if the "map" directory exists, if not, create it
        if not os.path.exists(img2_directory):
            os.makedirs(img2_directory)

        # Now you have both paths
        print(f"Video Path: {video_path}")
        print(f"Img2 Path: {img2_path}")

        return video_path, img2_path
    else:
        print("No file selected.")
        return None, None


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

def calculate_fovs(horizontal_fov_degrees, image_width, image_height):
    # Convert horizontal FOV from degrees to radians
    fov_x = math.radians(horizontal_fov_degrees)

    # Calculate vertical FOV using the aspect ratio of the image
    fov_y = 2 * math.atan(image_height / image_width * math.tan(fov_x / 2))

    return fov_x, fov_y


a = 0.1
b = -0.2
c = 0.001
d = 0.001
e = 0.0


def adjust_fov(frame, fov_x, fov_y, img_size):
    # Modify here with actual distortion coefficients from calibration
    dist_coeffs = np.array([a/100000,-b/100000,c/10000/100,d/10000/100,-e])  # Example distortion coefficients

    # Assuming the typo in 'img2.shape' is corrected to 'frame.shape'
    camera_matrix = fov_to_intrinsic(fov_x, fov_y, frame.shape[0], frame.shape[1])

    # Apply the transformation
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, img_size, 1, img_size)
    adjusted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image
    x, y, w, h = roi
    adjusted_frame = adjusted_frame[y:y + h, x:x + w]
    return adjusted_frame


def apply_perspective_transform():
    if len(points_list_img1) == 4 and len(points_list_img2) == 4:
        pts1 = np.float32(points_list_img1)
        pts2 = np.float32(points_list_img2)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_img = cv2.warpPerspective(original_frame_img1, matrix, (img2.shape[1], img2.shape[0]))
        cv2.imshow('Transformed Image', transformed_img)
        return transformed_img



fovx = calculate_fovs(10,1920,1080)
fovy = calculate_fovs(10,1920,1080)


# Now call the function to get the paths
video_path, img2_path = select_video_file()


# Load the second image
original_img2 = cv2.imread(img2_path)
img2 = original_img2.copy()

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Windows for display
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image2', cv2.WINDOW_NORMAL)
cv2.namedWindow('Image3', cv2.WINDOW_NORMAL)
cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)

cv2.setMouseCallback('Image1', click_handler.click_event_img1)
cv2.setMouseCallback('Image2', click_handler.click_event_img2)


# Create trackbars for FOV adjustment
cv2.createTrackbar('FOV X', 'Image', 10, 1000, handler.on_fov_changex)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('FOV Y', 'Image', 10, 1000, handler.on_fov_changey)  # Adjust the range as needed
# Create trackbars for FOV adjustment
cv2.createTrackbar('A', 'Image', 10, 10000, handler.on_A)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('B', 'Image', 10, 10000, handler.on_B)  # Adjust the range as needed
cv2.createTrackbar('C', 'Image', 10, 1000, handler.on_C)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('D', 'Image', 10, 1000, handler.on_D)  # Adjust the range as needed
cv2.createTrackbar('E', 'Image', 0, 1000, handler.on_E)  # Assuming FOV range from 0 to 180


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if  fovx==0:
        key = cv2.waitKey(1) & 0xFF
        continue

    if  fovy==0:
        key = cv2.waitKey(1) & 0xFF
        continue

    img2 = original_img2
    # Resize the frame to match img2's display size if needed
    frame_img1 = cv2.resize(frame, (frame.shape[1], frame.shape[0]))

    # Adjust FOV here (example: 90 degrees in X, 90 degrees in Y, adjust as needed)
    frame_img1 = adjust_fov(frame_img1, fovx/10, fovy/10, (frame_img1.shape[1], frame_img1.shape[0]))
    original_frame_img1 = frame_img1.copy()

    if len(points_list_img1) > 0:
        click_handler.draw_points(frame_img1, points_list_img1)

    cv2.imshow('Image1', frame_img1)

    cv2.imshow('Image2', img2)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    transformed_img = apply_perspective_transform()
    if transformed_img is not None:
    # Resize img2 to match img1's dimensions if necessary
      if transformed_img.shape[:2] != img2.shape[:2]:
            img2 = cv2.resize(img2, (transformed_img.shape[1], transformed_img.shape[0]))

      # Blend the images
      alpha = 0.5  # This represents the weight of the first image. 0.5 gives both images equal weight.
      beta = (1.0 - alpha)  # This represents the weight of the second image.
      gamma = 0  # Scalar added to each sum
      blended_image = cv2.addWeighted(transformed_img, alpha, img2, beta, gamma)

      cv2.imshow('Image3', blended_image)

cap.release()
cv2.destroyAllWindows()
