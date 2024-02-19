import cv2
import numpy
import numpy as np

class ImageResizer:
    def __init__(self):
        self.image = numpy.ndarray
        pass

    def ResizeWithAspectRatioAndFill(image, width=None, height=None, inter=cv2.INTER_AREA):
        # Initial dimension is None
        dim = None
        # Get the original dimensions of the image
        (h, w) = image.shape[:2]

        # If both width and height are None, return the original image
        if width is None and height is None:
            return image

        # Resize based on the height while maintaining aspect ratio
        if height is not None:
            r = height / float(h)
            dim = (int(w * r), height)

        # Resize the image with the new dimensions
        resized = cv2.resize(image, dim, interpolation=inter)

        # If width is also specified, adjust the image
        if width is not None and dim[0] < width:
            # Calculate the difference in width
            delta_w = width - dim[0]
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            # Create a new image with the specified width and height, filled with black
            new_image = np.zeros((height, width, 3), dtype=np.uint8)
            # Place the resized image in the center of the new image
            new_image[:, left:left + dim[0]] = resized
            return new_image
        else:
            return resized

