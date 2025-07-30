from config import MEDIA_PATH, PLAYLISTS_PATH
import os
from url_to_filename import filename_to_url


if __name__ == "__main__":
    # get a list of all files in MEDIA_PATH
    files = os.listdir(MEDIA_PATH)
    # do filename_to_url on each file name
    urls = [filename_to_url(os.path.splitext(file)[0]) for file in files]
    # filter out any URLs that are not valid (e.g., empty strings)
    urls = [url for url in urls if url]
    # get a list of all the URLs in all the playlists in PLAYLISTS_PATH
    playlist_files = os.listdir(PLAYLISTS_PATH)
    playlist_urls = []
    for playlist_file in playlist_files:
        with open(os.path.join(PLAYLISTS_PATH, playlist_file), 'r') as f:
            playlist_urls.extend(f.read().splitlines())
    # find the URLs that are in the playlists but not in MEDIA_PATH
    to_download = set(playlist_urls) - set(urls)
    # print the URLs to download
    if to_download:
        print("URLs to download:")
        for url in to_download:
            print(url)
    else:
        print("No URLs to download found.")
    # find the URLs that are in MEDIA_PATH but not in any playlist
    lost_urls = set(urls) - set(playlist_urls)
    # print the lost URLs
    if lost_urls:
        print("Lost URLs:")
        for url in lost_urls:
            print(url)
    else:
        print("No lost URLs found.")