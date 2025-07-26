import os
import pyperclip

PATH = r"C:\Users\N6506\Home\health\entertainment\news_underground\mediaSorter\programs\mediaSorter_Python\data\media\\"

def create_playlist(playlist_name, video_paths):
    playlist_dir = "playlists"
    os.makedirs(playlist_dir, exist_ok=True)
    playlist_path = os.path.join(playlist_dir, f"{playlist_name}.txt")

    with open(playlist_path, 'w', encoding='utf-8') as f:
        for path in video_paths:
            f.write(path.strip() + '\n')
    print(f"Playlist '{playlist_name}' saved to {playlist_path}")

def main():
    clipboard_text = pyperclip.paste()
    lines = clipboard_text.splitlines()

    if not lines:
        print("Clipboard is empty.")
        return

    playlist_name = lines[0]
    video_paths = [PATH + line for line in lines[3:] if line]

    if not playlist_name:
        print("Playlist name is empty!")
        return

    if not video_paths:
        print("No video URLs found!")
        return

    create_playlist(playlist_name, video_paths)

if __name__ == "__main__":
    main()
