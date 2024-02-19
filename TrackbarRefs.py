
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
