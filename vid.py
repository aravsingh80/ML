
# from datetime import timedelta
# import cv2
# import numpy as np
# import os
# from moviepy.editor import VideoFileClip
# SAVING_FRAMES_PER_SECOND = 10
# def format_timedelta(td):
#     """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
#     omitting microseconds and retaining milliseconds"""
#     result = str(td)
#     try:
#         result, ms = result.split(".")
#     except ValueError:
#         return (result + ".00").replace(":", "-")
#     ms = int(ms)
#     ms = round(ms / 1e4)
#     return f"{result}.{ms:02}".replace(":", "-")


# def main(video_file):
#     # load the video clip
#     video_clip = VideoFileClip(video_file)
#     # make a folder by the name of the video file
#     filename, _ = os.path.splitext(video_file)
#     filename += "-moviepy"
#     if not os.path.isdir(filename):
#         os.mkdir(filename)

#     # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
#     saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
#     # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
#     step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
#     # iterate over each possible frame
#     for current_duration in np.arange(0, video_clip.duration, step):
#         # format the file name and save it
#         frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration))
#         frame_filename = os.path.join(filename, f"frame{frame_duration_formatted}.jpg")
#         # save the frame with the current duration
#         video_clip.save_frame(frame_filename, current_duration)
# main("vid.mp4")
import cv2
import os

image_folder = 'vid-moviepy'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()