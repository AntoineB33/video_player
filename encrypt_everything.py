import os
from encrypt import encrypt_file
from config import ENCRYPTED_MEDIA_PATH, DECRYPTED_MEDIA_PATH

def encrypt_all_files():
    for file in os.listdir(DECRYPTED_MEDIA_PATH):
        decrypted_path = os.path.join(DECRYPTED_MEDIA_PATH, file)
        encrypted_path = os.path.join(ENCRYPTED_MEDIA_PATH, file)

        if os.path.isfile(decrypted_path):
            try:
                encrypt_file(decrypted_path, encrypted_path)
                print(f"Encrypted: {file}")
            except Exception as e:
                print(f"Failed to encrypt {file}: {e}")

if __name__ == "__main__":
    encrypt_all_files()