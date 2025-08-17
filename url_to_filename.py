import base64
import pyperclip

def url_to_filename(url: str) -> str:
    """
    Encode a URL to a Windows-safe unique filename.
    """
    encoded = base64.urlsafe_b64encode(url.encode('utf-8')).decode('ascii')
    return encoded.rstrip('=')  # Remove padding to make it more compact

def filename_to_url(filename: str) -> str:
    """
    Decode a Windows-safe filename back to the original URL.
    """
    padding_needed = 4 - (len(filename) % 4)
    if padding_needed != 4:
        filename += '=' * padding_needed  # Restore padding
    decoded = base64.urlsafe_b64decode(filename.encode('ascii')).decode('utf-8')
    return decoded

if __name__ == "__main__":
    url = pyperclip.paste()
    filename = url_to_filename(url)
    pyperclip.copy(filename)
