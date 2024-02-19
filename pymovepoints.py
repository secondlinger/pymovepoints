import math
import os
import cv2
import numpy as np
from ImageResizer import ImageResizer
from fileselector import VideoFileSelector
import pickle
import pyautogui

def create_aligned_points(start_point, end_point, distance_interval):
    # Convert points to numpy arrays for easier calculation
    start = np.array(start_point)
    end = np.array(end_point)

    # Calculate the vector from start to end
    vector = end - start

    # Calculate the distance between start and end
    total_distance = np.linalg.norm(vector)

    # Normalize the vector to get the direction
    direction = vector / total_distance

    # Calculate the number of points to generate
    num_points = int(total_distance // distance_interval)

    # Initialize the list of points with the start point
    points_list_10metInv = [start_point]

    # Generate each point
    for i in range(1, num_points):
        # Calculate the point's position
        point = start + direction * (distance_interval * i)
        # Append the point to the list
        points_list_10metInv.append(tuple(point))

     #Optionally, add the end point to the list
    points_list_10metInv.append(end_point)

    return points_list_10metInv

def get_filename_without_extension(path):
    # Split the path to get the filename with extension
    _, filename_with_extension = os.path.split(path)
    # Split the filename from its extension
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    return filename_without_extension

def SaveData():
    global points_list_img1, points_list_img2, points_list_10met, points_list_10metInv, TenMeterLen, fovx, fovy, a, b, c, d, e
    data_to_save = {
        'points_list_img1': points_list_img1,
        'points_list_img2': points_list_img2,
        'points_list_10met': points_list_10met,
        'points_list_10metInv': points_list_10metInv,
        'TenMeterLen': TenMeterLen,
        'fov_x': 10,  # Example value, replace with actual trackbar value
        'fov_y': 10,  # Example value, replace with actual trackbar value
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
    }

    global video_path
    filename = get_filename_without_extension(video_path)

    #make directory if not exist
    if not os.path.exists('data'):
        os.makedirs('data')


    with open("data/"+filename+'.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)

def load_data_from_file():
    data_to_save = {
        'points_list_img1': [],
        'points_list_img2': [],
        'points_list_10met': [],
        'points_list_10metInv': [],
        'TenMeterLen': 0,
        'fov_x': 10,  # Example value, replace with actual trackbar value
        'fov_y': 10,  # Example value, replace with actual trackbar value
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
    }

    global video_path
    filename = get_filename_without_extension(video_path)
    filename = "data/" + filename + '.pkl'

    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return data_to_save

# Initialize lists to store coordinates
points_list_img1 = []
points_list_img2 = []

points_list_10met  = []
points_list_10metInv  = []

TenMeterLen = 0

a = 0.1
b = -0.2
c = 0.001
d = 0.001
e = 0.0


def resize_points(points, original_size, new_size):
    """
    Resize points based on a change in window size.

    Parameters:
    - points: List of tuples containing (x, y) coordinates.
    - original_size: Tuple of (original_width, original_height).
    - new_size: Tuple of (new_width, new_height).

    Returns:
    - List of tuples with updated (x, y) coordinates.
    """
    original_width, original_height = original_size
    new_width, new_height = new_size

    # Calculate scaling factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Apply scaling to points
    resized_points = [(int(x * scale_x), int(y * scale_y)) for x, y in points]

    return resized_points

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

def resize_point(point, original_size, new_size):
    original_width, original_height = original_size
    new_width, new_height = new_size

    # Calculate scaling factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Apply scaling to points
    resized_points = (int(point[0] * scale_x), int(point[1] * scale_y))

    return resized_points

class ImageFunctions:
    def __init__(self):
        pass

    @staticmethod
    def draw_points(img, points_list, frame_name):
        for point in points_list:
            cv2.circle(img, point, 5, (255, 0, 0), -1)
        return img

    @staticmethod
    def click_event_img1(event, x, y, flags, params):
        global OriginImage
        handle_click_event(event, x, y, points_list_img1, OriginImage, 'Image1')

    @staticmethod
    def click_event_img2(event, x, y, flags, params):
        global MapImage
        handle_click_event(event, x, y, points_list_img2, MapImage, 'Image2')

    @staticmethod
    def click_event_img3(event, x, y, flags, params):
        global OriginalImage
        handle_click_event10met(event, x, y, points_list_10met, OriginalImage, 'Image2')

    @staticmethod
    def click_event_img4(event, x, y, flags, params):
        global OriginalToInvImage
        handle_click_event10met(event, x, y, points_list_10metInv, OriginalToInvImage, 'OriginalToInv')

    @staticmethod
    def draw_on(points_list, img):
        for point in points_list:
            cv2.circle(img, point, 5, (0,0 , 255), -1)
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
                global OriginImage
                OriginImage = img_copy
            else:
                global MapImage
                MapImage = img_copy
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
            if window_name == 'TenMeterRef':
                global OriginImage
                OriginImage = img_copy
            else:
                global MapImage
                MapImage = img_copy
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

# Calculate the distance between the two transformed points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def apply_perspective_transform_and_map_points(original_img,src_points, matrix):
        transformed_points = []
        for pt in src_points:
            point_homogenous = np.array([*pt, 1]).reshape(-1, 1)
            transformed_point = matrix.dot(point_homogenous)
            transformed_point = transformed_point / transformed_point[2]  # Normalize to convert from homogenous coordinates
            transformed_point = transformed_point[:2].astype(int).reshape(-1)  # Convert coordinates to integer
            transformed_points.append((transformed_point[0], transformed_point[1]))

        # Draw the src_points on the original image for reference
        for pt in src_points:
            cv2.circle(original_img, (int(pt[0]), int(pt[1])), 30, (0, 0, 0), -1)

        return OriginalToInvImage, transformed_points

def apply_inverse_perspective_transform_and_map_points(target_img, dest_points, matrix):
    # Invert the matrix for reverse transformation
    matrix_inv = np.linalg.inv(matrix)

    # Transform dest_points from original_img to target_img
    transformed_points = []
    for pt in dest_points:
            cv2.circle(target_img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            point_homogenous = np.array([*pt, 1]).reshape(-1, 1)
            transformed_point = matrix_inv.dot(point_homogenous)
            transformed_point = transformed_point / transformed_point[2]  # Normalize to convert from homogenous coordinates
            transformed_point = transformed_point[:2].astype(int).reshape(-1)  # Convert coordinates to integer
            transformed_points.append((transformed_point[0], transformed_point[1]))
            # Draw the transformed points on the original image

    return target_img, transformed_points

def apply_perspective_transform():
    if len(points_list_img1) == 4 and len(points_list_img2) == 4:
        pts1 = np.float32(points_list_img1)
        pts2 = np.float32(points_list_img2)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_img = cv2.warpPerspective(original_frame_img1, matrix, (MapImage.shape[1], MapImage.shape[0]))

        return transformed_img, matrix
    else:
        return None, None

def apply_inverse_perspective_transform():
    if len(points_list_img1) == 4 and len(points_list_img2) == 4:
        # Points from the second image (destination points)
        pts2 = np.float32(points_list_img2)
        # Points from the first image (source points)
        pts1 = np.float32(points_list_img1)

        # Compute the perspective transform matrix from img2 to img1
        matrix = cv2.getPerspectiveTransform(pts2, pts1)

        # Apply the inverse transformation to img2 to get back to the perspective of img1
        # Assuming `original_frame_img2` is the image you want to transform back and `img1` is the target perspective
        transformed_img = cv2.warpPerspective(OriginalToInvImage, matrix, (OriginalToInvImage.shape[1], OriginalToInvImage.shape[0]))

        return transformed_img, matrix
    else:
        return None, None

video_selector = VideoFileSelector()

# Now call the function to get the paths
video_path, img2_path = video_selector.select_video_file()

# Load the second image
original_img2 = cv2.imread(img2_path)
OriginalImage = cv2.imread(img2_path)
OriginalToInvImage = cv2.imread(img2_path)
MapImage = original_img2.copy()

# Initialize video capture
cap = cv2.VideoCapture(video_path)

data_loaded = load_data_from_file()

# Access your data, using get to provide default values if keys are missing
points_list_img1 = data_loaded.get('points_list_img1', [])
points_list_img2 = data_loaded.get('points_list_img2', [])
# Continue for other data as necessary
# Example of accessing trackbar values, providing default if not found
trackbar_values = data_loaded.get('trackbar_values', {})
fovx = trackbar_values.get('FOV X', 10)
fovy = trackbar_values.get('FOV Y', 10)

a = data_loaded.get('a', 0)
b = data_loaded.get('b', 0)
c = data_loaded.get('c', 0)
d = data_loaded.get('d', 0)
e = data_loaded.get('e', 0)

#----------------------------------------------------------------------------------
# Windows for display
cv2.namedWindow('Image')
cv2.namedWindow('Image1')
cv2.namedWindow('Image2')
cv2.namedWindow('OriginalToInv', cv2.WINDOW_NORMAL)

cv2.setMouseCallback('Image1', ImageFunctions.click_event_img1)
cv2.setMouseCallback('Image2', ImageFunctions.click_event_img2)
cv2.setMouseCallback('OriginalToInv', ImageFunctions.click_event_img4)

# Create trackbars for FOV adjustment
cv2.createTrackbar('FOV X', 'Image', fovx, 1000, TrackbarCallbacks.on_fov_change_x)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('FOV Y', 'Image', fovy, 1000, TrackbarCallbacks.on_fov_change_y)  # Adjust the range as needed

# Create trackbars for FOV adjustment
cv2.createTrackbar('A', 'Image', int(a), 10000, TrackbarCallbacks.on_A)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('B', 'Image', int(b), 10000, TrackbarCallbacks.on_B)  # Adjust the range as needed
cv2.createTrackbar('C', 'Image', int(c), 1000, TrackbarCallbacks.on_C)  # Assuming FOV range from 0 to 180
cv2.createTrackbar('D', 'Image', int(d), 1000, TrackbarCallbacks.on_D)  # Adjust the range as needed
cv2.createTrackbar('E', 'Image', int(e), 1000, TrackbarCallbacks.on_E)  # Assuming FOV range from 0 to 180

#----------------------------------------------------------------------------------

scale = 1
IsMouseMove = False
MouseMoveIndex = 0

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

    MapImage = original_img2

    # Resize the frame to match img2's display size if needed
    OriginImage = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale))
    MapImage = cv2.resize(MapImage, (MapImage.shape[1] * scale, MapImage.shape[0] * scale))

    MapImage = ImageResizer.ResizeWithAspectRatioAndFill(MapImage, width=1780, height=1000)

    frame_img2 = cv2.resize(MapImage, (MapImage.shape[1] * scale, MapImage.shape[0] * scale))

    OriginImage = adjust_fov(OriginImage, fovx / 10, fovy / 10, (OriginImage.shape[1], OriginImage.shape[0]))
    original_frame_img1 = OriginImage.copy()

    if len(points_list_img1) > 0:
        ImageFunctions.draw_points(OriginImage, points_list_img1, 'Image1')
    if len(points_list_img2) > 0:
        ImageFunctions.draw_points(frame_img2, points_list_img2, 'Image2')

    OriginalToInvImage, matrix = apply_perspective_transform()

    if matrix is not None:
        apply_perspective_transform_and_map_points(original_frame_img1, points_list_10metInv, matrix)

        if len(points_list_10metInv) == 2:
            TenMeterLen = calculate_distance(points_list_10metInv[0], points_list_10metInv[1])

        if(len(points_list_10metInv)>9):
            # Example usage
            start_point = points_list_10metInv[0]
            end_point = points_list_10metInv[len(points_list_10metInv)-1]
            distance_interval = TenMeterLen  # For example, 2 units of distance

            points_list = create_aligned_points(start_point, end_point, distance_interval)
            points_list_10metInv = points_list

        OriginalToInvImage, detransformed_points = apply_inverse_perspective_transform_and_map_points(OriginalToInvImage, points_list_10metInv, matrix)

    if OriginalToInvImage is not None:
        OriginImage = ImageFunctions.draw_points(OriginImage, detransformed_points, 'Image1')

        cv2.imshow('OriginalToInv', OriginalToInvImage)

        if OriginalToInvImage is not None:
            Calibration_image = blend_images(OriginalToInvImage, MapImage, 0.5)
            cv2.imshow('Calibration', Calibration_image)

    cv2.imshow('Image1', OriginImage)
    cv2.imshow('Image2', frame_img2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s'):
        SaveData()

    if key == ord('m'):
        IsMouseMove = not IsMouseMove
        MouseMoveIndex = 0

    if IsMouseMove:
        #detransformed_points
        windowsize = (1776, 707)
        basepos = (136, 287)
        if MouseMoveIndex >= len(detransformed_points):
            IsMouseMove = False
        else:
            resizedwindowpos = resize_point(detransformed_points[MouseMoveIndex], (OriginImage.shape[1], OriginImage.shape[0]), (windowsize[0], windowsize[1]))
            resizedwindowpos = (resizedwindowpos[0] + basepos[0], resizedwindowpos[1] + basepos[1])
            pyautogui.moveTo(resizedwindowpos[0], resizedwindowpos[1])
            pyautogui.click()
            MouseMoveIndex += 1


cap.release()
cv2.destroyAllWindows()
