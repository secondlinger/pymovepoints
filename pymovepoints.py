import math
import cv2
import numpy as np

# Initialize lists to store coordinates
points_list_img1 = []
points_list_img2 = []

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

# This function will remain mostly unchanged
def draw_points(img, points_list):
    for point in points_list:
        cv2.circle(img, point, 5, (255, 0, 0), -1)
    cv2.imshow('Image1' if img is frame_img1 else 'Image2', img)

# Updated to work with video frames
def click_event_img1(event, x, y, flags, params):
    global frame_img1
    handle_click_event(event, x, y, points_list_img1, frame_img1, 'Image1')

# This function stays the same
def click_event_img2(event, x, y, flags, params):
    global img2
    handle_click_event(event, x, y, points_list_img2, img2, 'Image2')

# No changes here
def handle_click_event(event, x, y, points_list, img, window_name):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points_list) < 4:
            points_list.append((x, y))
            draw_points(img, points_list)
            print(f"Point added to {window_name}: ({x},{y})")
        else:
            print(f"Maximum points reached for {window_name}.")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points_list:
            points_list.pop()
            img_copy = original_frame_img1.copy() if window_name == 'Image1' else original_img2.copy()
            draw_points(img_copy, points_list)
            if window_name == 'Image1':
                global frame_img1
                frame_img1 = img_copy
            else:
                global img2
                img2 = img_copy
            print(f"Last point removed from {window_name}.")

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

# Callback function for the trackbar
def on_fov_changex(trackbarValue):
    global fovx
    fovx = trackbarValue

def on_fov_changey(trackbarValue):
    global fovy
    fovy = trackbarValue

def on_A(trackbarValue):
    global a
    a = trackbarValue

def on_B(trackbarValue):
    global b
    b = trackbarValue

def on_C(trackbarValue):
    global c
    c = trackbarValue

def on_D(trackbarValue):
    global d
    d = trackbarValue

def on_E(trackbarValue):
    global e
    e = trackbarValue


# Video path
video_path = 'img1.mp4'  # Replace with the path to your video
img2_path = 'img1.png'  # Path to your second image

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

cv2.setMouseCallback('Image1', click_event_img1)
cv2.setMouseCallback('Image2', click_event_img2)


# Create trackbars for FOV adjustment
cv2.createTrackbar('FOV X', 'Image', 10, 1000, on_fov_changex)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('FOV Y', 'Image', 10, 1000, on_fov_changey)  # Adjust the range as needed
# Create trackbars for FOV adjustment
cv2.createTrackbar('A', 'Image', 10, 10000, on_A)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('B', 'Image', 10, 10000, on_B)  # Adjust the range as needed
cv2.createTrackbar('C', 'Image', 10, 1000, on_C)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('D', 'Image', 10, 1000, on_D)  # Adjust the range as needed
cv2.createTrackbar('E', 'Image', 0, 1000, on_E)  # Assuming FOV range from 0 to 180


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
        draw_points(frame_img1, points_list_img1)

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
