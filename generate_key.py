from cryptography.fernet import Fernet
from config import KEY_PATH

key = Fernet.generate_key()
with open(KEY_PATH, "wb") as f:
    f.write(key)
print(f"Saved key to {KEY_PATH}")
