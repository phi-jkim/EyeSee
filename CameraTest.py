import os
import time
from datetime import datetime
from picamera2 import Picamera2, Preview
from pprint import *


class MyCamera(Picamera2):

	def __init__(self, camera_id, directory, name):
		super().__init__(camera_id)
		self.directory = directory
		self.last_second = -1
		self.image_count = 0
		self.name = name


	def create_config(self):
		self.config = self.create_still_configuration(main={"size": (2592, 2592)}, raw={"size": (4608, 2592)})
	def print_config(self):
		print(self.name + " config: " + str(self.config["main"]))


	def capture_count_interval(self, count=1, interval=1):
		for i in range(count):
			current_time = datetime.now()
			current_second = current_time.second

			if current_second != self.last_second:
				self.image_count = 0
				self.last_second = current_second

			timestamp = current_time.strftime("%Y%m%d_%H%M%S")
			self.filename = f"{timestamp}_{self.image_count}_{self.name}.jpg"
			self.full_path = os.path.join(self.directory, self.filename)


			self.capture_file(self.full_path)
			self.image_count += 1

			time.sleep(interval)

	def captureBoth_count_interval(count=1, interval=1):
		last_second = -1
		image_count = 0
		directory = "/home/bread/Desktop/TestPics"

		while True:
			current_time = datetime.now()
			current_second = current_time.second

			if current_second != last_second:
				image_count = 0
				last_second = current_second

			timestamp = current_time.strftime("%Y%m%d_%H%M%S")
			filename = f"{timestamp}_{image_count}_RGB.jpg"
			full_path = os.path.join(directory, filename)
			cam0.capture_file(full_path)

			filename = f"{timestamp}_{image_count}_IR.jpg"
			full_path = os.path.join(directory, filename)
			cam1.capture_file(full_path)

			image_count += 1

			time.sleep(interval)


cam0 = MyCamera(0, "/home/bread/Desktop/TestPics", "RGB")
cam1 = MyCamera(1, "/home/bread/Desktop/TestPics", "IR")

'''
# <Config info>
	# GENERAL, global params
		* transform: hori/vert flip
		* queue
		* sensor
			output_size (res)
			bit_depth
			> will auto pick best match, use exact sensor mode spec for certainty
		...

	# STREAM-specific configs (main stream always defined)
		* image size
			> optimize using cam.align_configuration(config)
		* image format
			For main:
			• XBGR8888 - every pixel is packed into 32-bits, with a dummy 255 value at the end, so a pixel would look like [R, G, B,
			255] when captured in Python. (These format descriptions can seem counter-intuitive, but the underlying
			infrastructure tends to take machine endianness into account, which can mix things up!)
			• XRGB8888 - as above, with a pixel looking like [B, G, R, 255].
			• RGB888 - 24 bits per pixel, ordered [B, G, R].
			• BGR888 - as above, but ordered [R, G, B].
			• YUV420 - YUV images with a plane of Y values followed by a quarter plane of U values and then a quarter plane of V
			values.

	# SENSOR mode (res, bit depth, etc.) can be specified.
		> If not, will be inferred from raw stream (if present).
		* cam.sensor_modes
			bit_depth
			crop_limits (fov)
			exposure_limits
			format: can be passed as stream image format
			fps
			size (res)

	# RUNTIME controls excluded unless specified


# Config generation
config = cam.create_still_configuration

# Apply configuration
cam.configure(config)

# View configuration
cam.camera_configuration()

'''

sensor_modes = (cam0.sensor_modes)
sensor_mode = sensor_modes[2]
print(sensor_mode)

cam0.create_config()
cam1.create_config()
cam0.align_configuration(cam0.config)
cam1.align_configuration(cam0.config)
cam0.configure(cam0.config)
cam1.configure(cam1.config)
cam0.print_config()
cam1.print_config()

cam0.start()
cam1.start()

MyCamera.captureBoth_count_interval(0, 1)

cam0.close()
cam1.close()

# cam0.camera_configuration()
# cam0.camera_controls

# cam0.create_config()
# cam0.print_config()

# cam0.align_configuration(cam0.config)
# cam0.print_config()

# cam0.configure(cam0.config)

# # cam0.start_preview(Preview.QTGL)
# cam0.start(show_preview=True)

# cam0.capture_count_interval()

# cam0.close()
# cam1.close()