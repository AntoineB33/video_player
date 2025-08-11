import os
from encrypt import decrypt_file
from config import ENCRYPTED_MEDIA_PATH, DECRYPTED_MEDIA_PATH

def decrypt_all_files():
    for file in os.listdir(ENCRYPTED_MEDIA_PATH):
        encrypted_path = os.path.join(ENCRYPTED_MEDIA_PATH, file)
        decrypted_path = os.path.join(DECRYPTED_MEDIA_PATH, file)

        if os.path.isfile(encrypted_path):
            try:
                decrypt_file(encrypted_path, decrypted_path)
                print(f"Decrypted: {file}")
            except Exception as e:
                print(f"Failed to decrypt {file}: {e}")

if __name__ == "__main__":
    decrypt_all_files()