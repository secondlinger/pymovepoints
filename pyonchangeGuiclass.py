import cv2

class ImageClickHandler:
    def __init__(self):
        self.points_list_img1 = []
        self.points_list_img2 = []
        self.original_frame_img1 = None
        self.original_img2 = None
        self.frame_img1 = None
        self.img2 = None

    def set_original_images(self, frame_img1, img2):
        self.original_frame_img1 = frame_img1.copy()
        self.original_img2 = img2.copy()
        self.frame_img1 = frame_img1
        self.img2 = img2

    def draw_points(self, img, points_list):
        for point in points_list:
            cv2.circle(img, point, 5, (255, 0, 0), -1)
        cv2.imshow('Image1' if img is self.frame_img1 else 'Image2', img)

    def click_event_img1(self, event, x, y, flags, params):
        self.handle_click_event(event, x, y, self.points_list_img1, self.frame_img1, 'Image1')

    def click_event_img2(self, event, x, y, flags, params):
        self.handle_click_event(event, x, y, self.points_list_img2, self.img2, 'Image2')

    def handle_click_event(self, event, x, y, points_list, img, window_name):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points_list) < 4:
                points_list.append((x, y))
                self.draw_points(img, points_list)
                print(f"Point added to {window_name}: ({x},{y})")
            else:
                print(f"Maximum points reached for {window_name}.")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points_list:
                points_list.pop()
                img_copy = self.original_frame_img1.copy() if window_name == 'Image1' else self.original_img2.copy()
                self.draw_points(img_copy, points_list)
                if window_name == 'Image1':
                    self.frame_img1 = img_copy
                else:
                    self.img2 = img_copy
                print(f"Last point removed from {window_name}.")
