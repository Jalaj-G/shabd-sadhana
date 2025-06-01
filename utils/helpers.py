import os
import zipfile
from pathlib import Path
from typing import List, Tuple
from utils.logger import setup_logger
logger = setup_logger()

def extract_zip(zip_source, target_dir):
    """
    Accepts either a path or an open file object from Gradio,
    verifies it is a valid ZIP, and extracts to `target_dir`.
    """
    # 1) Turn file-like object into a path we control
    if hasattr(zip_source, "read"):           # this is a file object (e.g. gr.File)
        zip_source.seek(0)
        tmp_path = Path(zip_source.name)
    else:                                     # assume it's a str/Path
        tmp_path = Path(zip_source)

    if not zipfile.is_zipfile(tmp_path):
        raise ValueError(f"{tmp_path} is not a valid ZIP archive.")

    try:
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile as e:
        logger.error("Corrupted ZIP: %s", e)
        raise
    finally:
        # If you want to delete the temp file after use:
        # if hasattr(zip_source, "read"): tmp_path.unlink(missing_ok=True)
        pass

def get_audio_transcript_pairs(extracted_dir: str) -> List[Tuple[str, str]]:
    """
    Scans the extracted directory for audio and transcript files,
    pairing them based on matching filenames (excluding extensions).

    Returns:
        A list of tuples: (audio_file_path, transcript_file_path)
    """
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    transcript_extensions = {'.txt', '.srt'}

    audio_files = {}
    transcript_files = {}

    # Walk through the directory to find audio and transcript files
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            file_path = os.path.join(root, file)
            stem, ext = os.path.splitext(file)
            ext = ext.lower()

            if ext in audio_extensions:
                audio_files[stem] = file_path
            elif ext in transcript_extensions:
                transcript_files[stem] = file_path

    # Pair audio and transcript files based on matching stems
    paired_files = []
    for stem in audio_files:
        if stem in transcript_files:
            paired_files.append((audio_files[stem], transcript_files[stem]))

    return paired_files
