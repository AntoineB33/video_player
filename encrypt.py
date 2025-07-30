from cryptography.fernet import Fernet
from pathlib import Path
from config import KEY_PATH
import os

def load_key(key_path=KEY_PATH):
    return Path(key_path).read_bytes()

def encrypt_file(in_path, out_path, key_path=KEY_PATH):
    key = load_key(key_path)
    f = Fernet(key)
    data = Path(in_path).read_bytes()          # reads entire file
    token = f.encrypt(data)                    # encrypted + integrity-protected
    Path(out_path).write_bytes(token)
    os.remove(in_path)

def decrypt_file(in_path, out_path, key_path=KEY_PATH):
    key = load_key(key_path)
    f = Fernet(key)
    token = Path(in_path).read_bytes()
    data = f.decrypt(token)                    # verifies integrity before decrypting
    Path(out_path).write_bytes(data)
    os.remove(in_path)

# if __name__ == "__main__":
    # decrypt_file(
    #     in_path=Path(MEDIA_PATH) / "aHR0cHM6Ly9lY2NoaS5pd2FyYS50di92aWRlb3Mvbm1sZ3lpNDRuZnBqM2EyYQ.mp4",
    #     out_path=Path(MEDIA_PATH) / "example_decrypted.mp4"
    # )
    # Uncomment below to

    # folder_path = r"C:\Users\N6506\Home\health\entertainment\news_underground\mediaSorter\media"
    # # encrypt all files from folder_path to MEDIA_PATH
    # for file in Path(folder_path).glob("*.*"):
    #     in_path = file
    #     out_path = Path(MEDIA_PATH) / file.name
    #     encrypt_file(in_path, out_path)
    #     print(f"Encrypted {in_path} to {out_path}")
