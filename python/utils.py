from pathlib import Path
import re

def get_cv2_pattern_from_folder(folder_path):
    folder = Path(folder_path)
    files = sorted([f for f in folder.iterdir() if f.is_file()])

    if not files:
        raise ValueError("No files found in folder.")

    first_name = files[0].name
    # Match any prefix, then a sequence of digits, then extension
    match = re.search(r'^(.*?)(\d+)(\.\w+)$', first_name)

    if not match:
        raise ValueError(f"Could not extract numeric pattern from: {first_name}")

    prefix, num_part, ext = match.groups()
    pad_len = len(num_part)
    # Compose pattern with prefix, zero-padded number, and extension
    return f"{prefix}%0{pad_len}d{ext}"