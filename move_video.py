import os
import shutil
import pyperclip
import re
import sys
from url_to_filename import url_to_filename
from encrypt import encrypt_file
from config import ENCRYPTED_MEDIA_PATH

# === CONFIGURATION ===
destination_folder = ENCRYPTED_MEDIA_PATH  # Use config

# Read clipboard content
clipboard_content = pyperclip.paste()

file_path = sys.argv[1] if len(sys.argv) > 1 else None

if file_path and clipboard_content and os.path.isfile(file_path) and re.match(r'^(https?://)', clipboard_content):
    # Get the original file name and extension
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)

    new_name = url_to_filename(clipboard_content) + ext

    # Ensure destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # Full path to the new file
    new_path = os.path.join(destination_folder, new_name)

    if os.path.exists(new_path):
        input(f"File {new_name} already exists in {destination_folder}.")
    else:
    
        # Encrypt the file before moving
        encrypt_file(file_path, new_path)
else:
    input("Clipboard does not contain a valid quoted file path.")
