# createshorts.py
import os
import re
import yt_dlp
from faster_whisper import WhisperModel
import moviepy
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings
import requests
import json
import shutil
import subprocess
from tqdm import tqdm
import pickle
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Configure ImageMagick binary path
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# --- Configuration ---
DATA_DIR = os.path.abspath("data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
EDITED_VIDEOS_DIR = os.path.join(DATA_DIR, "edited-videos")
EDITED_SHORTS_DIR = os.path.join(DATA_DIR, "edited-shorted")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
COOKIES_FILE = "cookies.txt"

for dir_path in [DATA_DIR, VIDEOS_DIR, AUDIO_DIR, EDITED_VIDEOS_DIR, EDITED_SHORTS_DIR, TRANSCRIPT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Settings ---
subtitle_offset = 0
max_subtitle_duration = 60
subtitle_font = "Calibri-Bold"
subtitle_fontsize = 70
subtitle_color = "white"
subtitle_stroke_width = 4
subtitle_stroke_color = "black"
zoom_factor = 1.5
subtitle_vertical_align = 'bottom'
subtitle_vertical_offset = 550
shorts_duration = 52
CREATE_SHORTS = True

# --- Helper Functions ---
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

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
            raise RuntimeError("ffmpeg installed but not found in PATH.")
    except Exception as e:
        print(f"Failed to install ffmpeg: {e}")
        return False

def ensure_imagemagick_installed():
    if shutil.which("magick") is not None:
        print("ImageMagick is already installed.")
        return True
    print("ImageMagick is not installed.")
    print("Please install ImageMagick from https://imagemagick.org/script/download.php")
    input("Press Enter after you have installed ImageMagick and added it to your PATH...")
    if shutil.which("magick") is not None:
        print("ImageMagick installed successfully.")
        return True
    else:
        raise RuntimeError("ImageMagick installed but not found in PATH.")

def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

# --- Download Video ---
def download_video(url, cookies_file=COOKIES_FILE):
    ydl_opts_info = {'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id')
            video_title = info.get('title', 'video')
            video_title = sanitize_filename(video_title)
    except Exception as e:
        print(f"Error extracting video info: {e}")
        return None, None, None

    video_filename = f"{video_title} - {video_id}.mp4"
    video_file = os.path.join(VIDEOS_DIR, video_filename)
    if os.path.exists(video_file):
        print(f"Video {video_id} already exists. Reusing existing file.")
        return video_file, video_id, video_title

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': video_file,
        'quiet': True,
        'merge_output_format': 'mp4',
        'cookiefile': cookies_file,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id')
            video_title = info.get('title', 'video')
            video_title = sanitize_filename(video_title)
            video_filename = f"{video_title} - {video_id}.mp4"
            video_file = os.path.join(VIDEOS_DIR, video_filename)
            return video_file, video_id, video_title
    except Exception as e:
        print(f"Error downloading video with yt-dlp: {e}")
        return None, None, None

# --- Transcription Functions ---
def transcribe_audio(video_path, video_id, update_progress=None):
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}_audio.mp3")
    if os.path.exists(audio_path):
        print(f"Audio for video {video_id} already exists. Reusing existing audio file.")
    else:
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
            if update_progress:
                progress = (pbar.n / total_duration) * 100
                update_progress(progress, stage='transcribing')
    return transcript

def group_words_with_timestamps(segments, group_size=3):
    subtitle_groups = []
    for seg in segments:
        if hasattr(seg, "words") and seg.words:
            words = seg.words
            for i in range(0, len(words), group_size):
                group = words[i:i+group_size]
                start = group[0].start
                end = group[-1].end
                text = " ".join(w.word for w in group)
                subtitle_groups.append({"start": start, "end": end, "text": text})
        else:
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

def create_shorts(video_path, subtitle_groups, video_id, video_title, zoom_factor=1.0, task_id=None, update_progress=None):
    try:
        video = VideoFileClip(video_path)
        print(f"Video loaded. Size: {video.size}, Duration: {video.duration}")
    except Exception as e:
        print(f"Error opening video file: {e}")
        return None

    w, h = video.size
    target_width, target_height = 1080, 1920
    scaling_factor = min(target_width / w, target_height / h) * zoom_factor
    new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)
    video_resized = video.resize((new_w, new_h))

    background = ColorClip(size=(target_width, target_height), color=(0, 0, 0)).set_duration(video.duration)

    subtitle_clips = []
    for group in subtitle_groups:
        duration = group["end"] - group["start"]
        if duration <= 0:
            continue
        tc = TextClip(
            group["text"],
            font=subtitle_font,
            fontsize=subtitle_fontsize,
            color=subtitle_color,
            stroke_color=subtitle_stroke_color,
            stroke_width=subtitle_stroke_width,
            align='center',
            method='caption',
            size=(target_width - 40, None),
            kerning=-0.5,
        ).set_start(group["start"]).set_duration(duration)
        text_w, text_h = tc.size
        pos_y = {
            'bottom': target_height - text_h - subtitle_vertical_offset,
            'top': subtitle_vertical_offset,
            'center': (target_height - text_h) / 2 + subtitle_vertical_offset
        }.get(subtitle_vertical_align, target_height - text_h - subtitle_vertical_offset)
        pos_y = max(0, min(pos_y, target_height - text_h))
        tc = tc.set_position(('center', pos_y))
        subtitle_clips.append(tc)

    final_clip = CompositeVideoClip(
        [background, video_resized.set_position(((target_width - new_w) / 2, (target_height - new_h) / 2))] + subtitle_clips,
        size=(target_width, target_height)
    )

    edited_filename = f"{task_id}_{video_title} - {video_id}_edited.mp4" if task_id else f"{video_title} - {video_id}_edited.mp4"
    output_file = os.path.join(EDITED_VIDEOS_DIR, edited_filename)
    temp_file = os.path.join(EDITED_VIDEOS_DIR, f"temp_{edited_filename}")

    # Step 1: Write temporary file with MoviePy and progress
    print("Starting MoviePy rendering...")
    final_clip.write_videofile(
        temp_file,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        bitrate="8000k",
        preset="medium",
        ffmpeg_params=["-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"],
        logger='bar',  # MoviePy progress bar
        write_logfile=False,
        fps=video.fps,
        verbose=True,
    )
    print("MoviePy rendering complete. Temporary file created:", temp_file)

    # Step 2: Use FFmpeg for final encoding with tqdm progress bar
    print("Starting FFmpeg re-encoding...")
    cmd = [
        "ffmpeg", "-i", temp_file, "-c:v", "libx264", "-c:a", "aac", "-b:v", "8000k",
        "-threads", "4", "-preset", "medium", "-progress", "-", "-nostats", "-y",
        output_file
    ]
    env = os.environ.copy()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, env=env)
    duration = video.duration
    print(f"Video duration: {duration} seconds")

    with tqdm(total=100, desc="Rendering Video", unit="%") as pbar: # tqdm for FFmpeg progress
        last_progress = 0
        if update_progress:
            update_progress(0, stage='video_processing')
        for line in process.stdout:
            #print("FFmpeg output:", line.strip()) # Optional: Print FFmpeg output for debugging
            if "out_time=" in line:
                time_str = line.split("out_time=")[1].strip()
                try:
                    current_time = time_to_seconds(time_str)
                    progress = min((current_time / duration) * 100, 100)
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                        if update_progress:
                            update_progress(progress, stage='video_processing') # Call update_progress here!
                except ValueError:
                    print(f"Failed to parse time: {time_str}")
        process.wait()
        if last_progress < 100:
            pbar.update(100 - last_progress)
            if update_progress:
                update_progress(100, stage='video_processing')

    if os.path.exists(temp_file):
        os.remove(temp_file)

    print(f"Video saved to: {output_file}")
    final_clip.close()
    video.close()
    return output_file

def create_video_shorts(edited_video_path, video_id, video_title, segment_duration=52, task_id=None, update_progress=None):
    print("Creating video shorts...")
    output_pattern = os.path.join(EDITED_SHORTS_DIR, f"{task_id}_{video_title} - {video_id}_Part %01d.mp4" if task_id else f"{video_title} - {video_id}_Part %01d.mp4")
    cmd = [
        "ffmpeg", "-i", edited_video_path, "-c", "copy", "-map", "0",
        "-segment_time", str(segment_duration), "-f", "segment",
        "-reset_timestamps", "1", output_pattern
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    duration = VideoFileClip(edited_video_path).duration
    num_segments = int(duration // segment_duration) + (1 if duration % segment_duration else 0)
    if update_progress:
        update_progress(0, stage='shorts_creation')
    segment_count = 0
    for line in process.stdout:
        # Look for lines indicating a new segment file is being opened
        if "Opening" in line and ".mp4" in line:
            segment_count += 1
            if update_progress:
                progress = min((segment_count / num_segments) * 100, 100)
                update_progress(progress, stage='shorts_creation')
    process.wait()
    if update_progress:
        update_progress(100, stage='shorts_creation')
    shorts_files = [
        f for f in os.listdir(EDITED_SHORTS_DIR)
        if f.startswith(f"{task_id}_{video_title} - {video_id}_Part " if task_id else f"{video_title} - {video_id}_Part ") and f.endswith(".mp4")
    ]
    shorts_paths = [os.path.join(EDITED_SHORTS_DIR, f) for f in shorts_files]
    print(f"Video shorts created: {shorts_paths}")
    return shorts_paths

def get_transcript_path(video_id):
    return os.path.join(TRANSCRIPT_DIR, f"{video_id}_transcript.pkl")

def main(youtube_url, task_id=None, shorts_duration=52, cookies_file=COOKIES_FILE, update_progress=None):
    if not ensure_ffmpeg_installed() or not ensure_imagemagick_installed():
        raise RuntimeError("Required dependencies not installed.")

    video_path, video_id, video_title = download_video(youtube_url, cookies_file)
    if not video_path or not video_id:
        raise ValueError("Video download failed.")

    transcript_file = get_transcript_path(video_id)
    if os.path.exists(transcript_file):
        with open(transcript_file, 'rb') as f:
            transcript = pickle.load(f)
        # Immediately set transcription progress to 100% if transcript exists
        if update_progress:
            update_progress(100, stage='transcribing')
    else:
        transcript = transcribe_audio(video_path, video_id, update_progress)
        if transcript:
            with open(transcript_file, 'wb') as f:
                pickle.dump(transcript, f)
        else:
            raise RuntimeError("Transcription failed.")

    subtitle_groups = group_words_with_timestamps(transcript, group_size=3)
    edited_filename = f"{task_id}_{video_title} - {video_id}_edited.mp4" if task_id else f"{video_title} - {video_id}_edited.mp4"
    edited_video_path = os.path.join(EDITED_VIDEOS_DIR, edited_filename)

    if not os.path.exists(edited_video_path):
        edited_video_path = create_shorts(
            video_path, subtitle_groups, video_id, video_title,
            zoom_factor=1.5, task_id=task_id, update_progress=update_progress
        )
        if not edited_video_path:
            raise RuntimeError("Video rendering failed.")

    shorts_paths = create_video_shorts(
        edited_video_path, video_id, video_title,
        segment_duration=shorts_duration, task_id=task_id, update_progress=update_progress
    ) if shorts_duration > 0 else []

    return video_id, video_title, edited_video_path, shorts_paths

def process_uploaded_video_segment(video_path, task_id, from_time, to_time, update_progress=None):
    if not ensure_ffmpeg_installed() or not ensure_imagemagick_installed():
        raise RuntimeError("Required dependencies not installed.")

    video_id = f"uploaded_{task_id}"
    video_title = os.path.splitext(os.path.basename(video_path))[0]
    
    # Cut the video segment
    cut_video_path = cut_video_segment(video_path, video_id, from_time, to_time)
    
    # Transcribe the cut segment
    transcript_file = get_transcript_path(video_id)
    if os.path.exists(transcript_file):
        with open(transcript_file, 'rb') as f:
            transcript = pickle.load(f)
        if update_progress:
            update_progress(100, stage='transcribing')
    else:
        transcript = transcribe_audio(cut_video_path, video_id, update_progress)
        if transcript:
            with open(transcript_file, 'wb') as f:
                pickle.dump(transcript, f)
        else:
            raise RuntimeError("Transcription failed.")

    # Create subtitles and zoomed video
    subtitle_groups = group_words_with_timestamps(transcript, group_size=3)
    edited_filename = f"{task_id}_{video_title} - {video_id}_edited.mp4"
    edited_video_path = os.path.join(EDITED_VIDEOS_DIR, edited_filename)

    if not os.path.exists(edited_video_path):
        edited_video_path = create_shorts(
            cut_video_path, subtitle_groups, video_id, video_title,
            zoom_factor=1.5, task_id=task_id, update_progress=update_progress
        )
        if not edited_video_path:
            raise RuntimeError("Video rendering failed.")
    
    # Clean up the cut video file
    if os.path.exists(cut_video_path):
        os.remove(cut_video_path)

    return video_id, video_title, edited_video_path
def cut_video_segment(video_path, video_id, from_time, to_time):
    output_path = os.path.join(VIDEOS_DIR, f"{video_id}_cut.mp4")
    cmd = [
        "ffmpeg", "-i", video_path, "-ss", from_time, "-to", to_time,
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "libx264", "-c:a", "aac", "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    main(youtube_url)