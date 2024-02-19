class TrackbarHandler:
    def __init__(self):
        # Initialize your variables with default values
        self.fovx = 0
        self.fovy = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0

    def on_fov_changex(self, trackbarValue):
        self.fovx = trackbarValue

    def on_fov_changey(self, trackbarValue):
        self.fovy = trackbarValue

    def on_A(self, trackbarValue):
        self.a = trackbarValue

    def on_B(self, trackbarValue):
        self.b = trackbarValue

    def on_C(self, trackbarValue):
        self.c = trackbarValue

    def on_D(self, trackbarValue):
        self.d = trackbarValue

    def on_E(self, trackbarValue):
        self.e = trackbarValue
