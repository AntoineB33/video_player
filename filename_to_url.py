import pyperclip
from url_to_filename import filename_to_url

if __name__ == "__main__":
    clipboard_content = pyperclip.paste()
    res = filename_to_url(clipboard_content)
    pyperclip.copy(res)