import os
import yt_dlp
from faster_whisper import WhisperModel
import moviepy
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings
import re
import requests
import json
import shutil
import subprocess
from tqdm import tqdm
import pickle

# Configure ImageMagick binary path
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# --- Configuration ---
TRANSCRIPT_DIR = "transcripts"
VIDEOS_DIR = "videos"
AUDIO_DIR = "audio"
EDITED_VIDEOS_DIR = "edited-videos"  # New directory for edited videos
COOKIES_FILE = "cookies.txt"  # Place cookies.txt in same directory as this script

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(EDITED_VIDEOS_DIR, exist_ok=True) # Create edited-videos directory

# --- Subtitle Settings ---
subtitle_offset = 0            # Adjust subtitle start times if needed.
max_subtitle_duration = 60       # Maximum duration per subtitle clip

subtitle_font = "DilleniaUPC-Bold"   # Ensure this font is installed
subtitle_fontsize = 48
subtitle_color = "white"
subtitle_stroke_width = subtitle_fontsize / 10
subtitle_stroke_color = "white"
subtitle_position = ('center', 'bottom')

# --- Helper Functions ---
def ensure_ffmpeg_installed():
    if shutil.which("ffmpeg") is not None:
        print("ffmpeg is already installed.")
        return True
    print("Installing ffmpeg...")
    try:
        print("Please download ffmpeg from a reputable source and add it to your PATH.")
        input("Press Enter after you have installed FFmpeg and added it to your PATH...")
        if shutil.which("ffmpeg") is not None:
            print("ffmpeg installed successfully.")
            return True
        else:
            raise RuntimeError("ffmpeg installed but not found in PATH. Did you add it correctly?")
    except Exception as e:
        print(f"Failed to install ffmpeg: {e}")
        return False

def ensure_imagemagick_installed():
    if shutil.which("magick") is not None:
        print("ImageMagick is already installed.")
        return True
    print("ImageMagick is not installed.")
    print("Please install ImageMagick from https://imagemagick.org/script/download.php")
    print("Make sure to check 'Install legacy utilities (e.g. convert)' and add it to your PATH.")
    input("Press Enter after you have installed ImageMagick and added it to your PATH...")
    if shutil.which("magick") is not None:
        print("ImageMagick installed successfully.")
        return True
    else:
        raise RuntimeError("ImageMagick installed but not found in PATH. Did you add it correctly?")

def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

# --- Download Video and Subtitles ---
def download_video_and_subtitles(url, cookies_file=COOKIES_FILE):
    ydl_opts_info = {'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id')
    except Exception as e:
        print(f"Error extracting video info: {e}")
        return False, None, None

    video_file = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    if os.path.exists(video_file):
        print(f"Video {video_id} already exists. Reusing existing file.")
        captions_file = video_file.replace('.mp4', '.en.vtt')
        if not os.path.exists(captions_file):
            captions_file = video_file.replace('.mp4', '.en.srt')
        if os.path.exists(captions_file):
            os.rename(captions_file, 'captions.srt')
            has_captions = True
        else:
            has_captions = False
        return has_captions, video_file, video_id

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': video_file,
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'subtitle': '--embed-subs',
        'quiet': True,
        'merge_output_format': 'mp4',
        'cookiefile': cookies_file,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            subtitle_file = video_file.replace('.mp4', '.en.vtt')
            if not os.path.exists(subtitle_file):
                subtitle_file = video_file.replace('.mp4', '.en.srt')
            if os.path.exists(subtitle_file):
                os.rename(subtitle_file, 'captions.srt')
                return True, video_file, video_id
            return False, video_file, video_id
    except Exception as e:
        print(f"Error downloading video with yt-dlp: {e}")
        return False, None, None

# --- Transcription and Subtitle Parsing Functions ---
def transcribe_audio(video_path, video_id):
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}_audio.mp3")
    if os.path.exists(audio_path):
        print(f"Audio for video {video_id} already exists. Reusing existing audio file.")
    else:
        print("Loading Whisper model...")
        model_size = "base"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Extracting audio from video...")
        try:
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", "0",
                audio_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e.stderr.decode()}")
            return []

    print("Transcribing audio with progress bar...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    # Request word-level timestamps
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)

    transcript = []
    try:
        duration_command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        duration_output = subprocess.check_output(duration_command).decode("utf-8").strip()
        total_duration = float(duration_output)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting video duration: {e}")
        return []

    with tqdm(total=total_duration, desc="Transcribing", unit="s") as pbar:
        for seg in segments:
            transcript.append(seg)
            seg_duration = seg.end - seg.start
            pbar.update(seg_duration)
    return transcript

def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    segments = []
    i = 0
    while i < len(lines):
        if lines[i].isdigit():
            i += 1
            if i >= len(lines):
                break
            timestamp = lines[i].split(' --> ')
            if len(timestamp) < 2:
                i += 1
                continue
            start = timestamp[0].replace(',', '.')
            end = timestamp[1].replace(',', '.')
            i += 1
            if i >= len(lines):
                break
            text = lines[i]
            while i + 1 < len(lines) and lines[i + 1] and not lines[i + 1].isdigit():
                i += 1
                text += " " + lines[i]
            segments.append({
                "start": time_to_seconds(start),
                "end": time_to_seconds(end),
                "text": text,
                "words": None  # No word-level timestamps available
            })
        i += 1
    return segments

def group_words_with_timestamps(segments, group_size=3):
    """
    Group words from segments into subtitle groups of every group_size words.
    Uses actual word-level timestamps if available.
    """
    subtitle_groups = []
    for seg in segments:
        # Check if word-level timestamps are available in this segment
        if hasattr(seg, "words") and seg.words:
            words = seg.words  # List of word objects
            for i in range(0, len(words), group_size):
                group = words[i:i+group_size]
                # Use attribute access for word objects
                start = group[0].start
                end = group[-1].end
                text = " ".join(w.word for w in group)
                subtitle_groups.append({"start": start, "end": end, "text": text})
        elif seg.get("words"):
            # If segment is a dict with a "words" key (likely None), fallback to splitting text
            words = seg["text"].split()
            total_duration = seg["end"] - seg["start"]
            if words:
                word_duration = total_duration / len(words)
                for i in range(0, len(words), group_size):
                    group_words = words[i:i+group_size]
                    start = seg["start"] + i * word_duration
                    end = start + len(group_words) * word_duration
                    subtitle_groups.append({"start": start, "end": end, "text": " ".join(group_words)})
        else:
            # Fallback if no word-level timestamps are provided at all
            words = seg["text"].split()
            total_duration = seg["end"] - seg["start"]
            if words:
                word_duration = total_duration / len(words)
                for i in range(0, len(words), group_size):
                    group_words = words[i:i+group_size]
                    start = seg["start"] + i * word_duration
                    end = start + len(group_words) * word_duration
                    subtitle_groups.append({"start": start, "end": end, "text": " ".join(group_words)})
    return subtitle_groups

def create_shorts(video_path, subtitle_groups, video_id):
    try:
        # Load the original video
        video = VideoFileClip(video_path)
        print(f"Video loaded. Size: {video.size}, Duration: {video.duration}")
    except Exception as e:
        print(f"Error opening video file: {e}")
        return

    # Test reading a frame
    try:
        frame = video.get_frame(0)
        print(f"Successfully read frame at t=0. Shape: {frame.shape}")
    except Exception as e:
        print(f"Error reading frame: {e}")
        return

    # Step 1: Calculate scaling factor to fit within 1080x1920
    w, h = video.size
    target_width, target_height = 1080, 1920
    scaling_factor = min(target_width / w, target_height / h)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    print(f"Original size: ({w}, {h}), New size: ({new_w}, {new_h})")

    # Apply monkey-patch for Pillow compatibility (fixes 'ANTIALIAS' error)
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'):
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
        print("Applied Pillow monkey-patch: ANTIALIAS set to LANCZOS")

    # Resize the video while maintaining aspect ratio
    try:
        video_resized = video.resize((new_w, new_h))
        print("Video resized successfully.")
    except Exception as e:
        print(f"Error resizing video: {e}")
        return

    # Step 2: Create a white background clip for YouTube Shorts dimensions
    background = ColorClip(size=(target_width, target_height), color=(255, 255, 255)).set_duration(video.duration)

    # Step 3: Create subtitle clips with black background
    subtitle_clips = []
    subtitle_fontsize = 40  # Reduced from 48 to make subtitles smaller
    padding = 20  # Padding around text for the black box
    margin = 20   # Margin from the bottom of the video content

    for group in subtitle_groups:
        duration = group["end"] - group["start"]
        if duration <= 0:
            continue  # Skip invalid timings

        # Create the text clip
        tc = TextClip(
            group["text"],
            font=subtitle_font,
            fontsize=subtitle_fontsize,
            color=subtitle_color,
            stroke_width=subtitle_stroke_width,
            stroke_color=subtitle_stroke_color,
            align='center'
        )

        # Create a black background box slightly larger than the text
        bg = ColorClip(size=(tc.w + 2 * padding, tc.h + 2 * padding), color=(0, 0, 0))

        # Composite text on the black background, centered
        tc_on_bg = CompositeVideoClip([bg, tc.set_position('center')], size=bg.size)

        # Set timing for the subtitle clip
        tc_on_bg = tc_on_bg.set_start(group["start"]).set_duration(duration)

        # Position the subtitle at the bottom of the resized video content
        pos_y = new_h - tc_on_bg.h - margin
        if pos_y < 0:
            pos_y = 0  # Ensure it stays within bounds
        tc_on_bg = tc_on_bg.set_position(('center', pos_y))

        subtitle_clips.append(tc_on_bg)

    # Step 4: Composite the resized video with subtitles
    composite_video = CompositeVideoClip([video_resized] + subtitle_clips)

    # Step 5: Place the composite video centered on the white background
    position = ((target_width - new_w) / 2, (target_height - new_h) / 2)
    final_clip = CompositeVideoClip([background, composite_video.set_position(position)], size=(target_width, target_height))

    # Step 6: Write the output video
    output_file = os.path.join(EDITED_VIDEOS_DIR, f"{video_id}_edited.mp4")
    try:
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac", threads=4)
        print(f"Video saved to: {output_file}")
    except Exception as e:
        print(f"Error writing video file {output_file}: {e}")
    finally:
        # Clean up
        final_clip.close()
        video.close()
def get_transcript_path(video_id):
    return os.path.join(TRANSCRIPT_DIR, f"{video_id}_transcript.pkl")

def main(youtube_url, use_gemini=False, cookies_file=COOKIES_FILE):
    if not ensure_ffmpeg_installed():
        print("Exiting due to ffmpeg installation failure.")
        return
    if not ensure_imagemagick_installed():
         print("Exiting due to ImageMagick installation failure.")
         return

    has_captions, video_path, video_id = download_video_and_subtitles(youtube_url, cookies_file=cookies_file)
    if video_path is None or video_id is None:
        print("Exiting due to download failure.")
        return

    transcript_file = get_transcript_path(video_id)
    if os.path.exists(transcript_file):
        print(f"Loading existing transcript for video {video_id}...")
        with open(transcript_file, 'rb') as f:
            transcript = pickle.load(f)
    else:
        if has_captions:
            transcript = parse_srt('captions.srt')
        else:
            transcript = transcribe_audio(video_path, video_id)
        if transcript:
            with open(transcript_file, 'wb') as f:
                pickle.dump(transcript, f)
            print(f"Transcript saved for video {video_id}")
        else:
            print("Transcription failed. Exiting.")
            return

    # Group words into subtitle segments with proper timestamps
    subtitle_groups = group_words_with_timestamps(transcript, group_size=3)
    create_shorts(video_path, subtitle_groups, video_id) # Pass video_id to create_shorts

    # Cleanup temporary captions file
    for file in ["captions.srt"]:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file {file}: {e}")

    print("Created video with dynamic subtitles based on word-level timestamps.")

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")  # e.g., https://www.youtube.com/watch?v=6zkL91LzCMc https://www.youtube.com/watch?v=usrl2FUWhEE
    use_gemini = False  # No Gemini API Key needed.
    main(youtube_url, use_gemini=use_gemini)