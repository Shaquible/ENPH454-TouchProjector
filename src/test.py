import cv2
from pyusbcameraindex import enumerate_usb_video_devices_windows

# List the devices.
devices = enumerate_usb_video_devices_windows()
for device in devices:
    print(f"{device.index} == {device.name} (VID: {
          device.vid}, PID: {device.pid}, Path: {device.path}")
