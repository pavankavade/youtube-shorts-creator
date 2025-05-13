import os
import shutil
import subprocess
import tempfile
import uuid
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

import whisper # openai-whisper
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field # Import Pydantic

# --- Configuration ---
# Adjust IMAGEMAGICK_PATH if needed, although not directly used in this flow
IMAGEMAGICK_PATH = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
FFMPEG_PATH = "ffmpeg" # Assumes ffmpeg is in PATH, otherwise provide full path
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
PROCESSED_VIDEOS_DIR = BASE_DIR / "processed_videos"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(exist_ok=True)
PROCESSED_VIDEOS_DIR.mkdir(exist_ok=True)

# --- Simple State for Cleanup ---
current_temp_video_path_for_cleanup: Optional[Path] = None

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/processed_videos", StaticFiles(directory=PROCESSED_VIDEOS_DIR), name="processed_videos")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Pydantic Models ---
class SubtitleSegment(BaseModel):
    start: float
    end: float
    text: str

class FinalizeRequest(BaseModel):
    temp_video_path: str
    segments: List[SubtitleSegment]
    font_size: int
    font_name: str
    font_color: str
    outline_color: str
    outline_width: int
    shadow_offset: int
    position: int # Base alignment
    # x_offset and y_offset removed
    output_suffix: str = ".mp4"

# --- Helper Functions ---

def convert_html_color_to_ass(html_color: str) -> str:
    """Converts #RRGGBB to &HBBGGRR& (ASS format with alpha)"""
    if html_color.startswith('#'):
        html_color = html_color[1:]
    if len(html_color) == 6:
        r, g, b = html_color[0:2], html_color[2:4], html_color[4:6]
        return f"&H00{b}{g}{r}&" # Opaque
    return "&H00FFFFFF&" # Default to opaque white if format is wrong

def format_time_srt(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm"""
    if seconds < 0: seconds = 0.0
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3600000; milliseconds %= 3600000
    minutes = milliseconds // 60000; milliseconds %= 60000
    seconds = milliseconds // 1000; milliseconds %= 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def create_subtitle_segments(
    whisper_result: Dict[str, Any],
    words_per_segment: int
) -> List[Dict[str, Any]]:
    print(f"--- Inside create_subtitle_segments (words_per_segment={words_per_segment}) ---")
    new_segments = []
    original_segments = whisper_result.get("segments", [])

    if not original_segments:
        print("WARNING: No segments found in Whisper result.")
        return []

    if words_per_segment <= 0:
        print("Mode: Using original Whisper segments.")
        for segment_index, segment in enumerate(original_segments):
            if segment.get('text','').strip() and 'start' in segment and 'end' in segment and segment['end'] >= segment['start']:
                new_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
            else:
                 print(f"[Original Segment {segment_index}] Skipping invalid original segment: {segment}")
    else:
        print(f"Mode: Splitting segments into chunks of {words_per_segment} words.")
        total_words_processed = 0
        segments_processed = 0

        for segment_index, segment in enumerate(original_segments):
            segments_processed += 1
            words_in_segment = segment.get("words")

            if not words_in_segment:
                print(f"[Segment {segment_index}] WARNING: Word timestamps requested but NOT FOUND. Falling back to full segment.")
                if segment.get('text','').strip() and 'start' in segment and 'end' in segment and segment['end'] >= segment['start']:
                    new_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip()
                    })
                continue

            num_words = len(words_in_segment)
            chunks_in_this_segment = 0
            for i in range(0, num_words, words_per_segment):
                chunk = words_in_segment[i : i + words_per_segment]
                if not chunk: continue

                if 'start' not in chunk[0] or not isinstance(chunk[0]['start'], (int, float)) \
                   or 'end' not in chunk[-1] or not isinstance(chunk[-1]['end'], (int, float)):
                     print(f"  [Chunk {i//words_per_segment}] WARNING: Skipping chunk due to missing/invalid start/end time in words: {chunk}")
                     continue

                start_time = chunk[0]['start']
                end_time = chunk[-1]['end']
                text = " ".join([word.get('word', '?').strip() for word in chunk])

                if end_time >= start_time and text:
                    new_segments.append({'start': start_time, 'end': end_time, 'text': text})
                    chunks_in_this_segment += 1
                    total_words_processed += len(chunk)
                else:
                     print(f"  [Chunk {i//words_per_segment}] WARNING: Skipping invalid chunk: start={start_time}, end={end_time}, text='{text}'")
        print(f"--- Finished processing {segments_processed} original segments. Total words in new segments: {total_words_processed} ---")

    print(f"--- Exiting create_subtitle_segments, returning {len(new_segments)} segments. ---")
    return new_segments

def write_srt_file(segments: List[SubtitleSegment], srt_path: Path):
    print(f"--- Writing {len(segments)} segments to SRT file: {srt_path} ---")
    with open(srt_path, "w", encoding="utf-8") as f:
        count = 0
        for i, segment in enumerate(segments):
            if segment.start <= segment.end and segment.text.strip():
                f.write(f"{count + 1}\n")
                f.write(f"{format_time_srt(segment.start)} --> {format_time_srt(segment.end)}\n")
                cleaned_text = " ".join(segment.text.strip().split())
                f.write(f"{cleaned_text}\n\n")
                count += 1
            else:
                 print(f"   [SRT Write] Skipping invalid segment at index {i}: start={segment.start}, end={segment.end}, text='{segment.text}'")
    print(f"--- Finished writing SRT file. Wrote {count} valid segments. ---")


# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe/")
async def transcribe_video(
    video_file: UploadFile = File(...),
    whisper_model_name: str = Form("tiny"),
    words_per_segment: int = Form(0)
):
    global current_temp_video_path_for_cleanup
    print(f"\n--- /transcribe START ---")
    print(f"Received model={whisper_model_name}, words_per_segment={words_per_segment}")
    temp_video_path = None

    if current_temp_video_path_for_cleanup and current_temp_video_path_for_cleanup.exists():
        print(f"Cleaning up previous temp video: {current_temp_video_path_for_cleanup}")
        try:
            current_temp_video_path_for_cleanup.unlink()
        except Exception as e_clean:
            print(f"Warning: Could not delete previous temp video {current_temp_video_path_for_cleanup}: {e_clean}")
    current_temp_video_path_for_cleanup = None

    try:
        suffix = Path(video_file.filename).suffix
        safe_stem = "".join(c for c in Path(video_file.filename).stem if c.isalnum() or c in ('_','-'))
        safe_filename = f"{safe_stem}{suffix}" if safe_stem else f"upload{suffix}"
        temp_video_fd, temp_video_path_str = tempfile.mkstemp(suffix=safe_filename, prefix="vid_", dir=UPLOADS_DIR)
        temp_video_path = Path(temp_video_path_str)
        print(f"Saving uploaded video to new temporary file: {temp_video_path}")

        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        os.close(temp_video_fd)
        current_temp_video_path_for_cleanup = temp_video_path

        result = {}
        try:
            print(f"Loading Whisper model: {whisper_model_name}")
            model = whisper.load_model(whisper_model_name)
            request_word_timestamps = words_per_segment > 0
            print(f"Requesting word timestamps from Whisper: {request_word_timestamps}")
            print(f"Transcribing video: {temp_video_path}")
            result = model.transcribe(
                str(temp_video_path),
                fp16=False,
                word_timestamps=request_word_timestamps
            )
            print("Whisper transcription finished.")
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {e}")

        print("Calling create_subtitle_segments...")
        processed_segments = create_subtitle_segments(result, words_per_segment)

        if not processed_segments:
             print("WARNING: No valid subtitle segments generated by create_subtitle_segments.")

        print(f"--- /transcribe END - Returning {len(processed_segments)} segments ---")
        return JSONResponse(content={
            "segments": processed_segments,
            "temp_video_path": str(temp_video_path),
            "output_suffix": suffix
        })

    except HTTPException as e:
        if temp_video_path and temp_video_path.exists():
             print(f"Cleaning up temp video due to HTTP error in /transcribe: {temp_video_path}")
             try: temp_video_path.unlink()
             except OSError: pass
             current_temp_video_path_for_cleanup = None
        raise e
    except Exception as e:
        print(f"An unexpected error occurred in /transcribe: {e}")
        import traceback
        traceback.print_exc()
        if temp_video_path and temp_video_path.exists():
             print(f"Cleaning up temp video due to unexpected error in /transcribe: {temp_video_path}")
             try: temp_video_path.unlink()
             except OSError: pass
             current_temp_video_path_for_cleanup = None
        raise HTTPException(status_code=500, detail=f"Unexpected server error during transcription: {type(e).__name__}")


@app.post("/finalize_video/")
async def finalize_video(request: Request, data: FinalizeRequest):
    print(f"\n--- /finalize_video START ---")
    print(f"Received {len(data.segments)} segments for video: {data.temp_video_path}")
    # Adjusted print statement as x_offset and y_offset are removed from data
    print(f"Style: Font={data.font_name}, Size={data.font_size}, Pos={data.position}")


    temp_srt_path = None
    processed_video_path = None
    temp_video_path = Path(data.temp_video_path)

    try:
        resolved_temp_path = temp_video_path.resolve()
        resolved_uploads_dir = UPLOADS_DIR.resolve()
        if resolved_uploads_dir not in resolved_temp_path.parents:
             raise ValueError("Attempt to access file outside of upload directory.")
        if not resolved_temp_path.exists():
             raise FileNotFoundError("Temporary video file not found.")
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="Temporary video file not found. Please start over by transcribing again.")
    except ValueError as ve:
         print(f"Security Warning: Invalid temporary video path. {ve}")
         raise HTTPException(status_code=400, detail="Invalid temporary video path provided.")
    except Exception as e_path:
         print(f"Error validating temporary path: {e_path}")
         raise HTTPException(status_code=500, detail="Server error validating video path.")

    try:
        temp_srt_fd, temp_srt_path_str = tempfile.mkstemp(suffix=".srt", prefix="sub_", dir=UPLOADS_DIR)
        temp_srt_path = Path(temp_srt_path_str)
        write_srt_file(data.segments, temp_srt_path)
        os.close(temp_srt_fd)
        print(f"Final SRT file generated: {temp_srt_path}")

        output_filename = f"captioned_{uuid.uuid4().hex}{data.output_suffix}"
        processed_video_path = PROCESSED_VIDEOS_DIR / output_filename
        safe_font_name = data.font_name.replace("'", "").replace('"', '')

        # --- Define Margins ---
        base_horizontal_margin = 20  # Default horizontal padding from video edges
        default_vertical_margin_from_edge = 30 # Default vertical distance from the relevant edge (top/bottom/middle-offset)

        # Specific vertical margin for "Bottom Center" (position 2)
        # This value makes it sit higher from the absolute bottom edge.
        # For Alignment=2 (BottomCenter), MarginV is distance from bottom edge.
        # A larger value pushes it further up.
        bottom_center_specific_vertical_margin = 70

        final_margin_l_value = base_horizontal_margin
        final_margin_r_value = base_horizontal_margin

        if data.position == 2:  # Bottom Center
            final_margin_v_value = bottom_center_specific_vertical_margin
            print(f"Using specific MarginV={final_margin_v_value} for Bottom Center (position 2).")
        else:
            # For other positions (top, other bottom, middle), use the default.
            # Note on ASS MarginV for Middle Alignments (4,5,6):
            # MarginV is often distance from the vertical center (0 = centered, positive = down).
            # So, default_vertical_margin_from_edge (e.g., 30) would push middle text 30px down.
            final_margin_v_value = default_vertical_margin_from_edge
            print(f"Using default MarginV={final_margin_v_value} for position {data.position}.")
        # --- End Margin Definition ---

        style_options = [
            f"FontName={safe_font_name}",
            f"FontSize={data.font_size}",
            f"PrimaryColour={convert_html_color_to_ass(data.font_color)}",
            f"OutlineColour={convert_html_color_to_ass(data.outline_color)}",
            f"BorderStyle=1", # 1 = Outline + Shadow
            f"Outline={data.outline_width}",
            f"Shadow={data.shadow_offset}",
            f"Alignment={data.position}", # Numpad alignment
            f"MarginL={final_margin_l_value}",
            f"MarginR={final_margin_r_value}",
            f"MarginV={final_margin_v_value}"
        ]
        force_style_str = ",".join(style_options)
        print(f"ASS Force Style String: {force_style_str}")

        srt_path_str = str(temp_srt_path.resolve())
        escaped_srt_path = srt_path_str.replace('\\', '/').replace(':', '\\:')
        subtitles_filter = f"subtitles=filename='{escaped_srt_path}':force_style='{force_style_str}'"
        print(f"Subtitles Filter String: {subtitles_filter}")

        ffmpeg_cmd = [
            FFMPEG_PATH, "-y",
            "-i", str(temp_video_path),
            "-vf", subtitles_filter,
            "-c:a", "copy",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            str(processed_video_path)
        ]
        print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")

        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        full_stderr = stderr.decode('utf-8', errors='ignore')

        if process.returncode != 0:
            print(f"--- FFmpeg stderr START ---\n{full_stderr}\n--- FFmpeg stderr END ---")
            lines = [line.strip() for line in full_stderr.splitlines() if line.strip()]
            specific_error = f"FFmpeg processing failed. Last error: {lines[-1]}" if lines else "FFmpeg processing failed."
            if processed_video_path.exists():
                try: processed_video_path.unlink()
                except OSError: pass
            raise HTTPException(status_code=500, detail=specific_error)
        else:
             if full_stderr.strip():
                 print(f"--- FFmpeg stderr (Success run) START ---\n{full_stderr}\n--- FFmpeg stderr END ---")

        print(f"Video processed successfully: {processed_video_path}")
        video_url = request.url_for('processed_videos', path=output_filename)
        print(f"--- /finalize_video END ---")
        return JSONResponse(content={"video_url": str(video_url), "message": "Video processed successfully."})

    except HTTPException as e:
        if temp_srt_path and temp_srt_path.exists():
             try: temp_srt_path.unlink()
             except OSError: pass
        raise e
    except Exception as e:
        print(f"An unexpected error occurred in /finalize_video: {e}")
        import traceback
        traceback.print_exc()
        if temp_srt_path and temp_srt_path.exists():
             try: temp_srt_path.unlink()
             except OSError: pass
        raise HTTPException(status_code=500, detail=f"Unexpected server error during finalization: {type(e).__name__}")
    finally:
        print("--- Cleaning up temporary SRT file (Finalize) ---")
        if temp_srt_path and temp_srt_path.exists():
            try:
                temp_srt_path.unlink()
                print(f"Cleaned up temp SRT: {temp_srt_path}")
            except Exception as e_clean:
                 print(f"Error cleaning up temp SRT {temp_srt_path}: {e_clean}")
        print(f"Keeping temp video for potential regeneration: {temp_video_path}")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Video Captioner Application ---")
    # ... (rest of the main execution block remains the same) ...
    print(f"Base Directory: {BASE_DIR}")
    print(f"IMAGEMAGICK_PATH set to: {IMAGEMAGICK_PATH}")
    if not Path(IMAGEMAGICK_PATH).exists() and IMAGEMAGICK_PATH != "magick":
         print(f"Warning: ImageMagick path does not exist: {IMAGEMAGICK_PATH}")

    try:
        print(f"Checking FFmpeg at: {FFMPEG_PATH}")
        ff_version_proc = subprocess.run([FFMPEG_PATH, "-version"], capture_output=True, check=True, text=True, encoding='utf-8', errors='ignore')
        print(f"FFmpeg found: {ff_version_proc.stdout.splitlines()[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: FFmpeg not found or not executable ('{FFMPEG_PATH}'). Please ensure it's installed and in PATH, or FFMPEG_PATH is set correctly. Error: {e}")
        exit(1)
    except Exception as e_ff:
         print(f"ERROR: An unexpected error occurred checking FFmpeg: {e_ff}")
         exit(1)

    try:
        print("Attempting to load 'tiny' Whisper model to check installation...")
        model_check = whisper.load_model("tiny")
        print("Whisper seems to be installed correctly.")
        del model_check
        import gc
        gc.collect()
    except Exception as e:
        print(f"ERROR: Whisper model could not be loaded. Ensure 'openai-whisper' is installed correctly and any dependencies (like PyTorch/torchvision/torchaudio and potentially CUDA for GPU) are met. Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"Uploads Dir: {UPLOADS_DIR}")
    print(f"Processed Videos Dir: {PROCESSED_VIDEOS_DIR}")
    print(f"Serving application on http://127.0.0.1:8000 (or http://0.0.0.0:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)