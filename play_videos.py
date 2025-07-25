import os
import glob

# Sort video files by modification time (newest first)
media_path = r"C:\Users\N6506\Home\health\entertainment\news_underground\mediaSorter\programs\mediaSorter_Python\data\media"
videos = sorted(glob.glob(os.path.join(media_path, "*.mp4"))) + sorted(glob.glob(os.path.join(media_path, "*.mkv")))  # Add other formats
videos.sort(key=os.path.getmtime, reverse=True)

# Play videos sequentially using default player
for video in videos:
    os.startfile(video)