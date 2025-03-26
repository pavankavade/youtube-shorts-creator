import os
from faster_whisper import WhisperModel
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings
import subprocess
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Configure ImageMagick
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# Directories
DATA_DIR = os.path.abspath("data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
EDITED_VIDEOS_DIR = os.path.join(DATA_DIR, "edited-videos")
EDITED_SHORTS_DIR = os.path.join(DATA_DIR, "edited-shorted")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
for dir_path in [DATA_DIR, VIDEOS_DIR, EDITED_VIDEOS_DIR, EDITED_SHORTS_DIR, AUDIO_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Settings
subtitle_font = "Calibri-Bold"
subtitle_fontsize = 70
subtitle_color = "white"
subtitle_stroke_width = 4
subtitle_stroke_color = "black"
zoom_factor = 1.5
subtitle_vertical_offset = 550

def transcribe_audio(video_path, video_id):
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}_audio.mp3")
    if not os.path.exists(audio_path):
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "libmp3lame", "-q:a", "0", audio_path
        ], check=True)
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    return list(segments)

def group_words_with_timestamps(segments, group_size=3):
    subtitle_groups = []
    for seg in segments:
        if hasattr(seg, "words") and seg.words:
            words = seg.words
            for i in range(0, len(words), group_size):
                group = words[i:i + group_size]
                subtitle_groups.append({
                    "start": group[0].start,
                    "end": group[-1].end,
                    "text": " ".join(w.word for w in group)
                })
    return subtitle_groups

def process_video(video_path, edited_filename, skip_editing=False):
    transcript = transcribe_audio(video_path, os.path.splitext(os.path.basename(video_path))[0])
    
    if skip_editing:
        return transcript  # Return only transcript if skipping editing
    
    video = VideoFileClip(video_path)
    w, h = video.size
    target_width, target_height = 1080, 1920
    scaling_factor = min(target_width / w, target_height / h) * zoom_factor
    new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)
    video_resized = video.resize((new_w, new_h))
    background = ColorClip(size=(target_width, target_height), color=(0, 0, 0)).set_duration(video.duration)

    subtitle_groups = group_words_with_timestamps(transcript)
    subtitle_clips = []
    for group in subtitle_groups:
        duration = group["end"] - group["start"]
        if duration <= 0:
            continue
        tc = TextClip(
            group["text"], font=subtitle_font, fontsize=subtitle_fontsize, color=subtitle_color,
            stroke_color=subtitle_stroke_color, stroke_width=subtitle_stroke_width, align='center',
            method='caption', size=(target_width - 40, None), kerning=-0.5
        ).set_start(group["start"]).set_duration(duration)
        text_w, text_h = tc.size
        pos_y = target_height - text_h - subtitle_vertical_offset
        tc = tc.set_position(('center', pos_y))
        subtitle_clips.append(tc)

    final_clip = CompositeVideoClip(
        [background, video_resized.set_position(((target_width - new_w) / 2, (target_height - new_h) / 2))] + subtitle_clips,
        size=(target_width, target_height)
    )
    output_file = os.path.join(EDITED_VIDEOS_DIR, edited_filename)
    
    # Set keyframe interval to 1 second based on video fps
    fps = video.fps  # Get the frame rate of the original video
    keyframe_interval = int(fps)  # Keyframe every 1 second (e.g., 30 for 30 fps)
    final_clip.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        bitrate="8000k",
        ffmpeg_params=["-g", str(keyframe_interval)]
    )
    
    final_clip.close()
    video.close()
    return output_file, transcript