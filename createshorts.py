import os
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings
import subprocess
import PIL.Image
import pysrt # Import pysrt
import math  # For ceiling function
import logging # For logging
import webvtt # Import webvtt
from datetime import timedelta # Import timedelta for VTT parsing

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Configure ImageMagick (Update path if necessary)
try:
    # Example for Windows, adjust if needed for other OS
    imagemagick_path = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
    if os.path.exists(imagemagick_path):
        change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
    else:
        # Try common Linux/macOS paths or rely on system PATH
        if os.path.exists("/usr/bin/magick"):
            change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/magick"})
        elif os.path.exists("/usr/local/bin/magick"):
            change_settings({"IMAGEMAGICK_BINARY": "/usr/local/bin/magick"})
        else:
            print("Warning: ImageMagick binary not found at expected paths. Text rendering might use default.")
except Exception as e:
    print(f"Warning: Could not set ImageMagick path. Text rendering might be affected. Error: {e}")

# Directories (ensure consistency with app.py)
DATA_DIR = os.path.abspath("data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
EDITED_VIDEOS_DIR = os.path.join(DATA_DIR, "edited-videos")
EDITED_SHORTS_DIR = os.path.join(DATA_DIR, "edited-shorts")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
SUBTITLES_DIR = os.path.join(DATA_DIR, "subtitles") # Ensure this exists
for dir_path in [DATA_DIR, VIDEOS_DIR, EDITED_VIDEOS_DIR, EDITED_SHORTS_DIR, AUDIO_DIR, SUBTITLES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Settings
subtitle_font = "Calibri-Bold" # Consider a more universally available font like Arial-Bold or specify path to font file
subtitle_fontsize = 60 # Slightly smaller maybe better with wrapping
subtitle_color = "white"
subtitle_stroke_width = 3 # Slightly smaller maybe better with wrapping
subtitle_stroke_color = "black"
# zoom_factor = 2.0 # Default zoom factor removed, now passed as argument
subtitle_vertical_offset = 550 # May need adjustment after wrapping
words_per_line = 4 # Max words per subtitle line

logger = logging.getLogger(__name__) # Use logger

def transcribe_audio(video_path, video_filename_base):
    """Extracts audio and transcribes using FasterWhisper."""
    audio_path = os.path.join(AUDIO_DIR, f"{video_filename_base}_audio.mp3")
    if not os.path.exists(audio_path):
        logger.info(f"Extracting audio from {video_path} to {audio_path}")
        try:
            subprocess.run([
                "ffmpeg", "-i", video_path, "-vn", # No video
                "-acodec", "libmp3lame", "-q:a", "2", # Good quality MP3
                "-y", # Overwrite if exists
                audio_path
            ], check=True, capture_output=True, text=True) # Capture output for errors
        except subprocess.CalledProcessError as e:
             logger.error(f"FFmpeg audio extraction failed (Code {e.returncode}): {e.stderr}")
             raise # Re-raise the exception
    logger.info(f"Transcribing audio file: {audio_path}")
    # Consider making model size configurable
    model_size = "base" # or "tiny", "small", "medium", "large-v2", etc.
    compute_type = "int8" # "float16" or "float32" for GPU, "int8" for CPU efficiency
    device = "cpu" # or "cuda" if NVIDIA GPU and CUDA libraries are installed
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        # Segments will contain word timestamps if word_timestamps=True
        segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
        logger.info(f"Transcription detected language: {info.language} with probability {info.language_probability}")
        return list(segments) # Return segments as a list
    except Exception as e:
         logger.error(f"Error during Whisper transcription: {e}", exc_info=True)
         raise

def group_words_with_timestamps(segments, group_size=words_per_line):
    """Groups transcribed words into subtitle lines with approx group_size words."""
    subtitle_groups = []
    current_line = []
    current_start = None

    for seg in segments:
        # Ensure seg.words exists and is iterable
        if hasattr(seg, "words") and seg.words:
            for word in seg.words:
                 # Ensure word has start, end, word attributes
                if not all(hasattr(word, attr) for attr in ['start', 'end', 'word']):
                    logger.warning(f"Skipping invalid word object: {word}")
                    continue

                if current_start is None:
                    current_start = word.start
                current_line.append(word)

                # If the line reaches the desired size, create a subtitle group
                if len(current_line) >= group_size:
                    start_time = current_start
                    end_time = word.end
                    text = " ".join(w.word.strip() for w in current_line)

                    if start_time is not None and end_time is not None and end_time > start_time:
                         subtitle_groups.append({
                            "start": start_time,
                            "end": end_time,
                            "text": text
                         })
                    else:
                         logger.warning(f"Skipping word group due to invalid times: Start={start_time}, End={end_time}, Text={text[:30]}...")

                    # Reset for the next line
                    current_line = []
                    current_start = None

    # Add any remaining words as the last subtitle line
    if current_line:
        start_time = current_start
        # Ensure the last word object is valid
        if current_line[-1] and hasattr(current_line[-1], 'end'):
             end_time = current_line[-1].end
             text = " ".join(w.word.strip() for w in current_line)
             if start_time is not None and end_time is not None and end_time > start_time:
                  subtitle_groups.append({
                      "start": start_time,
                      "end": end_time,
                      "text": text
                  })
             else:
                  logger.warning(f"Skipping final word group due to invalid times: Start={start_time}, End={end_time}, Text={text[:30]}...")
        else:
             logger.warning(f"Could not add final word group due to invalid last word object: {current_line[-1]}")


    logger.info(f"Grouped transcript into {len(subtitle_groups)} subtitle lines.")
    return subtitle_groups

def reformat_subtitle_text(text, max_words=words_per_line):
    """Wraps text to have max_words per line."""
    if not text or not isinstance(text, str):
        return ""
    words = text.split()
    lines = []
    for i in range(0, len(words), max_words):
        lines.append(" ".join(words[i:i + max_words]))
    return "\n".join(lines)

def parse_srt(filepath):
    """Parses an SRT file and reformats text."""
    logger.info(f"Parsing SRT file: {filepath}")
    subtitle_groups = []
    try:
        subs = pysrt.open(filepath, encoding='utf-8-sig') # Use utf-8-sig to handle potential BOM
    except Exception as e:
        logger.error(f"Error opening or parsing SRT file {filepath}: {e}", exc_info=True)
        # Try with plain utf-8 as fallback
        try:
            subs = pysrt.open(filepath, encoding='utf-8')
        except Exception as e2:
            logger.error(f"Fallback SRT parsing with utf-8 also failed for {filepath}: {e2}", exc_info=True)
            return [] # Return empty list on error

    for sub in subs:
        try:
            start_time = sub.start.ordinal / 1000.0  # Convert ms to seconds
            end_time = sub.end.ordinal / 1000.0    # Convert ms to seconds
            if start_time >= end_time:
                 logger.warning(f"Skipping SRT entry with invalid time: Start={start_time}, End={end_time}, Index={sub.index}")
                 continue
            # Clean and reformat text
            cleaned_text = sub.text.replace('\r', '').strip() # Remove CR, keep LF for potential structure, strip ends
            # If SRT contains specific line breaks, reformat_subtitle_text might override them.
            # Consider if preserving SRT line breaks is desired. For now, reformat based on word count.
            formatted_text = reformat_subtitle_text(cleaned_text.replace('\n', ' '), max_words=words_per_line) # Replace internal newlines before reformatting
            if formatted_text: # Only add if text is not empty
                subtitle_groups.append({
                    "start": start_time,
                    "end": end_time,
                    "text": formatted_text,
                    "raw_text": cleaned_text.replace('\n', ' ') # Store the cleaned, single-line text
                })
        except Exception as e:
            logger.error(f"Error processing SRT entry index {sub.index}: {e}. Text: '{sub.text[:50]}...'", exc_info=True)
            continue # Skip problematic entry

    logger.info(f"Parsed {len(subtitle_groups)} valid entries from SRT file.")
    return subtitle_groups

def parse_vtt(filepath):
    """Parses a VTT file and reformats text."""
    logger.info(f"Parsing VTT file: {filepath}")
    subtitle_groups = []
    try:
        # Ensure webvtt library correctly parses time strings to seconds
        vtt = webvtt.read(filepath, encoding='utf-8-sig') # Use utf-8-sig
    except Exception as e:
        logger.error(f"Error opening or parsing VTT file {filepath}: {e}", exc_info=True)
        # Try plain utf-8 as fallback
        try:
             vtt = webvtt.read(filepath, encoding='utf-8')
        except Exception as e2:
             logger.error(f"Fallback VTT parsing with utf-8 also failed for {filepath}: {e2}", exc_info=True)
             return []

    for caption in vtt:
        try:
            # webvtt-py properties start_in_seconds and end_in_seconds provide floats
            start_time = caption.start_in_seconds
            end_time = caption.end_in_seconds

            if start_time >= end_time:
                 logger.warning(f"Skipping VTT caption due to invalid time: Start={start_time}, End={end_time}, Text={caption.text[:50]}...")
                 continue

            # Clean and reformat text (reuse existing function)
            cleaned_text = caption.text.replace('\r', '').strip() # Remove CR, keep LF for potential structure, strip ends
            # Like SRT, reformat based on word count, replacing internal newlines
            formatted_text = reformat_subtitle_text(cleaned_text.replace('\n', ' '), max_words=words_per_line)

            if formatted_text: # Only add if text is not empty
                subtitle_groups.append({
                    "start": start_time,
                    "end": end_time,
                    "text": formatted_text,
                    "raw_text": cleaned_text.replace('\n', ' ') # Store the cleaned, single-line text
                })
            else:
                logger.warning(f"Skipping VTT caption due to empty formatted text. Original: {caption.text[:50]}...")

        except Exception as e:
             logger.error(f"Error processing VTT caption: {e}. Text: '{caption.text[:50]}...'", exc_info=True)
             continue # Skip problematic entry

    logger.info(f"Parsed {len(subtitle_groups)} valid captions from VTT file.")
    return subtitle_groups

# --- New Helper Function ---
def get_text_from_segments(segments):
    """Extracts concatenated raw text from parsed subtitle segments."""
    if not segments or not isinstance(segments, list):
        logger.warning("Invalid or empty segments received for text extraction.")
        return ""

    all_text = []
    for i, seg in enumerate(segments):
        if isinstance(seg, dict) and "raw_text" in seg and seg["raw_text"]:
            all_text.append(str(seg["raw_text"]).strip())
        elif isinstance(seg, dict) and "text" in seg and seg["text"]: # Fallback to formatted text if raw_text missing
            all_text.append(str(seg["text"]).replace('\n', ' ').strip())
        else:
            logger.warning(f"Segment {i} missing expected text field or is empty: {seg}")

    return " ".join(all_text) # Join with spaces
# --- End New Helper Function ---


# --- Modified process_video ---
def process_video(video_path, edited_filename, skip_editing=False, subtitle_file_path=None, zoom_factor=2.0):
    """
    Processes video: transcodes, optionally adds subtitles (generated or from file).

    Args:
        video_path (str): Path to the original video file.
        edited_filename (str): Desired filename for the output video.
        skip_editing (bool): If True, only performs transcription/parsing, no video editing.
        subtitle_file_path (str, optional): Path to an SRT or VTT subtitle file to use instead of generating.
        zoom_factor (float, optional): The factor by which to zoom into the video (e.g., 1.5, 2.0). Defaults to 2.0.

    Returns:
        tuple: (output_file_path, transcript_data, subtitle_groups_data)
               output_file_path is the path to the edited video (even if skip_editing, it's the expected path).
               transcript_data is the list of Whisper segments (or None if using external subtitles or transcription failed).
               subtitle_groups_data is the list of parsed/grouped subtitles (from file or Whisper, or None on error).
    """
    video_filename_base = os.path.splitext(os.path.basename(video_path))[0]
    transcript_data = None # Whisper segments list
    subtitle_groups = [] # Parsed/Grouped data for embedding or text extraction
    output_file = os.path.join(EDITED_VIDEOS_DIR, edited_filename) # Define potential output path early

    # Validate zoom factor
    try:
        zoom_factor = float(zoom_factor)
        if zoom_factor < 1.0: raise ValueError("Zoom factor must be >= 1.0")
    except (ValueError, TypeError):
        logger.warning(f"Invalid zoom factor '{zoom_factor}'. Using default 2.0.")
        zoom_factor = 2.0

    # 1. Determine Subtitle Source and Get Transcript/Subtitle Data
    processed_subtitle_file = False # Flag to track if we used an external file
    if subtitle_file_path and os.path.exists(subtitle_file_path):
        logger.info(f"Using provided subtitle file: {subtitle_file_path}")
        file_ext = os.path.splitext(subtitle_file_path)[1].lower()
        if file_ext == '.srt':
            subtitle_groups = parse_srt(subtitle_file_path)
            processed_subtitle_file = True
        elif file_ext == '.vtt':
            subtitle_groups = parse_vtt(subtitle_file_path)
            processed_subtitle_file = True
        else:
            logger.warning(f"Unsupported subtitle file extension: {file_ext}. Falling back to Whisper transcription.")
            # Fall through to transcription

        if processed_subtitle_file:
             transcript_data = None # No Whisper transcript generated
             if not subtitle_groups:
                 logger.warning(f"Parsing subtitle file {subtitle_file_path} resulted in empty data. Video will have no subtitles.")
        # If file wasn't found or wasn't a valid type, processed_subtitle_file is False, so we proceed to next block

    # Only run transcription if no valid subtitle file path was successfully processed
    if not processed_subtitle_file:
        logger.info("Attempting to generate transcript using Whisper.")
        try:
            # We might need transcript even if skip_editing is True (e.g., for Gemini suggestions later)
            transcript_data = transcribe_audio(video_path, video_filename_base)
            # Group words for burning subtitles or for text extraction
            subtitle_groups = group_words_with_timestamps(transcript_data)
            if not subtitle_groups:
                 logger.warning("Whisper transcription resulted in empty subtitle groups.")
        except Exception as e:
             logger.error(f"Transcription failed: {e}", exc_info=True)
             transcript_data = None # Ensure transcript is None on failure
             subtitle_groups = [] # Ensure no subtitles are attempted

    # 2. Handle skip_editing case
    if skip_editing:
        logger.info("Skipping video editing process.")
        # Even if skipping editing, return the transcript data and subtitle groups if generated/parsed
        return output_file, transcript_data, subtitle_groups

    # 3. Process Video with MoviePy (if not skipping)
    logger.info(f"Starting video processing with MoviePy for: {video_path}")
    video = None # Initialize video object variable
    subtitle_clips = [] # Initialize here
    try:
        video = VideoFileClip(video_path)
        # Validate video duration and size
        if not video.duration or video.duration <= 0:
            raise ValueError(f"Video file {video_path} has invalid duration: {video.duration}")
        if not video.size or video.size[0] <= 0 or video.size[1] <= 0:
             raise ValueError(f"Video file {video_path} has invalid dimensions: {video.size}")

        # --- Cropping/Resizing Logic ---
        w, h = video.size
        target_width, target_height = 1080, 1920 # 9:16 aspect ratio

        # Calculate scaling factor using MIN to fit, then zoom
        scaling_factor = min(target_width / w, target_height / h) * zoom_factor # <-- USES zoom_factor variable
        # Use int casting as per the old code reference
        new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)

        # Ensure new dimensions are valid
        if new_w <= 0 or new_h <= 0:
            # Log a warning or raise error - using current error handling style
            raise ValueError(f"Calculated invalid resize dimensions ({new_w}x{new_h})")

        video_resized = video.resize((new_w, new_h))

        # Create a black background clip with the target size
        background = ColorClip(size=(target_width, target_height), color=(0, 0, 0)).set_duration(video.duration) # <-- Creates background

        # NO EXPLICIT CROPPING HERE

        # video_resized = video_resized.set_duration(video.duration) # Duration is usually handled by background/composite

        # --- Subtitle Creation ---
        subtitle_clips = []
        if subtitle_groups:
            logger.info(f"Creating {len(subtitle_groups)} TextClips for subtitles.")
            for i, group in enumerate(subtitle_groups):
                # Ensure start/end are floats and duration is positive
                try:
                    start = float(group["start"])
                    end = float(group["end"])
                    duration = end - start
                except (ValueError, TypeError, KeyError) as e:
                     logger.warning(f"Skipping subtitle group {i} due to invalid time data: {group}. Error: {e}")
                     continue

                if duration <= 0:
                    logger.warning(f"Skipping subtitle group {i} due to zero or negative duration ({duration}): {group}")
                    continue
                # Use 'text' field which is formatted for embedding
                if "text" not in group or not group["text"]:
                    logger.warning(f"Skipping subtitle group {i} due to missing or empty formatted text: {group}")
                    continue

                try:
                    # Ensure text is a string before passing to TextClip
                    text_content_for_clip = str(group["text"]) # Already formatted by parser/grouper

                    if not text_content_for_clip: # Skip if formatting results in empty string
                         logger.warning(f"Skipping subtitle group {i} as formatted text is empty.")
                         continue

                    # Use target_width for caption sizing, similar to old code's (target_width - 40)
                    tc = TextClip(
                        text_content_for_clip, # Use pre-formatted text
                        font=subtitle_font, # Ensure font is valid/available on the system
                        fontsize=subtitle_fontsize,
                        color=subtitle_color,
                        stroke_color=subtitle_stroke_color,
                        stroke_width=subtitle_stroke_width,
                        align='center',
                        method='caption',
                        size=(target_width * 0.9, None), # Adjusted from (target_width - 40)
                        kerning=-0.5 # Optional adjustment
                    ).set_start(start).set_duration(duration)

                    # Position calculation (remains similar, based on target_height)
                    text_w, text_h = tc.size
                    pos_y = max(0, min(target_height - text_h, target_height - text_h - subtitle_vertical_offset))

                    tc = tc.set_position(('center', pos_y))
                    subtitle_clips.append(tc)
                except Exception as e:
                     # Log error creating specific TextClip, but continue with others
                     logger.error(f"Error creating TextClip for group {i}: {group}. Error: {e}", exc_info=True)


        # --- Composite and Write Video ---
        # Composite the background, the centered resized video, and subtitles
        final_clip = CompositeVideoClip(
            [background, video_resized.set_position(((target_width - new_w) / 2, (target_height - new_h) / 2))] + subtitle_clips, # <-- Uses background and positions video_resized
            size=(target_width, target_height)
        )
        # Ensure final duration matches original video (important as background sets duration)
        final_clip = final_clip.set_duration(video.duration)

        fps = video.fps if video.fps and video.fps > 0 else 30 # Ensure valid FPS
        keyframe_interval = int(fps * 2) # Keyframe every 2 seconds (adjust as needed)

        logger.info(f"Writing final video to: {output_file}")
        temp_audio_path = os.path.join(AUDIO_DIR, f"{video_filename_base}_temp_audio.aac")

        # ***** MODIFIED CALL *****
        final_clip.write_videofile(
            output_file,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            threads=os.cpu_count() or 4, # Use available cores or default
            fps=fps,
            preset='medium', # Balance speed and quality
            bitrate="5000k", # Adjust bitrate based on quality needs
            ffmpeg_params=[
                "-g", str(keyframe_interval), # Keyframe interval
                "-pix_fmt", "yuv420p",      # Pixel format for compatibility
                "-profile:v", "high",       # H.264 profile
                "-level:v", "4.1",          # H.264 level
                "-movflags", "+faststart",   # Optimize for web streaming
            ],
            logger='bar', # <-- ADDED THIS LINE FOR PROGRESS
            # verbose=False # Keep verbose False unless debugging ffmpeg directly
        )
        # ***** END OF MODIFICATION *****

        logger.info(f"Successfully wrote video file: {output_file}")

    except Exception as e:
         logger.error(f"MoviePy processing failed for {video_path}: {e}", exc_info=True)
         # Attempt to clean up potentially incomplete file
         if os.path.exists(output_file):
             try: os.remove(output_file)
             except OSError: logger.warning(f"Could not delete potentially incomplete output file: {output_file}")
         # Re-raise the exception to signal failure upstream
         raise

    finally:
        # --- Cleanup ---
        try:
            # Close clips if they were successfully created
            if 'final_clip' in locals() and final_clip: final_clip.close()
            if 'video_resized' in locals() and video_resized: video_resized.close()
            if video: video.close()
            for tc in subtitle_clips: tc.close()
            logger.debug("Closed MoviePy clips.")
        except Exception as e:
            logger.warning(f"Error during MoviePy cleanup: {e}")

    return output_file, transcript_data, subtitle_groups # Return path, whisper data, and parsed/grouped subs