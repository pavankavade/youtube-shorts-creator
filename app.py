from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
import threading
import os
import json
import subprocess
from werkzeug.utils import secure_filename
from createshorts import process_video, VIDEOS_DIR, EDITED_VIDEOS_DIR, EDITED_SHORTS_DIR
from google import genai
from google.genai import types
import logging

app = Flask(__name__, static_url_path='', static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///shorts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Models
class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(512), nullable=False)
    edited_filename = db.Column(db.String(512))
    status = db.Column(db.String(20), default='pending')
    video_title = db.Column(db.String(512))
    transcript = db.Column(db.Text)  # New field to store the transcript

class ShortSegment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    short_name = db.Column(db.String(100), nullable=False)  # New field: shortname
    short_description = db.Column(db.Text, nullable=False)  # Renamed to short_description
    start_time = db.Column(db.String(8), nullable=False)    # hh:mm:ss
    end_time = db.Column(db.String(8), nullable=False)      # hh:mm:ss
    short_filename = db.Column(db.String(512))
    status = db.Column(db.String(20), default='pending')

with app.app_context():
    db.create_all()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper Functions
def format_transcript(transcript):
    formatted = ""
    for seg in transcript:
        start = seg.start
        end = seg.end
        text = seg.text
        start_str = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02}"
        end_str = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02}"
        formatted += f"[{start_str} - {end_str}] {text}\n"
    return formatted

def get_suggested_segments(transcript):
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    
    client = genai.Client(api_key=api_key)
    prompt = (
        "Analyze the following video transcript and suggest engaging segments"
        "Each segment should be more than 50 seconds long but should be less than 60 seconds\n\n"
        "If a segment is longer than 1 minute, break it into multiple parts as part 1, part 2. Return the result as a JSON array where each object contains: "
        "'shortname' (a short, unique name for the clip), 'shortdescription' (a brief description), 'starttime' (in hh:mm:ss format), and 'endtime' (in hh:mm:ss format).\n\n" + transcript
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=8000,
            temperature=0.1
        )
    )
    logger.info(f"Gemini API raw response: {response.text}")
    # Strip Markdown code block syntax
    cleaned_response = response.text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]  # Remove ```json
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]  # Remove ```
    cleaned_response = cleaned_response.strip()  # Remove any remaining whitespace
    logger.info(f"Cleaned Gemini response: {cleaned_response}")
    return cleaned_response

def parse_segments(response_text):
    try:
        segments = json.loads(response_text)
        if not isinstance(segments, list):
            raise ValueError("Expected a JSON array")
        for seg in segments:
            if not all(k in seg for k in ['shortname', 'shortdescription', 'starttime', 'endtime']):
                raise ValueError("Missing required fields in segment")
            # Normalize key names here if needed
            seg['short_name'] = seg.pop('shortname')  # Convert 'shortname' to 'short_name'
            seg['short_description'] = seg.pop('shortdescription')
            seg['start_time'] = seg.pop('starttime')
            seg['end_time'] = seg.pop('endtime')
        logger.info(f"Parsed segments: {segments}")
        return segments
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return []
    except ValueError as e:
        logger.error(f"Invalid segment format: {e}")
        return []
    
def process_uploaded_video(video_id):
    with app.app_context():
        video = Video.query.get(video_id)
        video.status = 'processing'
        db.session.commit()
        try:
            original_video_path = os.path.join(VIDEOS_DIR, video.original_filename)
            edited_filename = f"{os.path.splitext(video.original_filename)[0]}_edited.mp4"
            edited_video_path = os.path.join(EDITED_VIDEOS_DIR, edited_filename)

            if os.path.exists(edited_video_path):
                logger.info(f"Edited video {edited_filename} already exists, skipping processing.")
                transcript = process_video(original_video_path, edited_filename, skip_editing=True)
                if not video.edited_filename:
                    video.edited_filename = edited_filename
            else:
                edited_video_path, transcript = process_video(original_video_path, edited_filename)
                video.edited_filename = edited_filename

            video.video_title = os.path.splitext(video.original_filename)[0]
            video.transcript = format_transcript(transcript)  # Save the formatted transcript
            db.session.commit()

            formatted_transcript = format_transcript(transcript)
            response_text = get_suggested_segments(formatted_transcript)
            segments = parse_segments(response_text)

            for seg in segments:
                segment = ShortSegment(
                    video_id=video_id,
                    short_name=seg['short_name'],
                    short_description=seg['short_description'],
                    start_time=seg['start_time'],
                    end_time=seg['end_time'],
                    status='pending'
                )
                db.session.add(segment)
            video.status = 'completed'
            db.session.commit()
            logger.info(f"Video {video_id} processing completed successfully.")
        except Exception as e:
            video.status = 'failed'
            logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
            db.session.commit()

def process_short(video_id, short_id):
    with app.app_context():
        video = Video.query.get(video_id)
        short = ShortSegment.query.get(short_id)
        try:
            short.status = 'processing'
            db.session.commit()
            if not video.edited_filename:
                raise ValueError("Edited video not found")
            edited_video_path = os.path.join(EDITED_VIDEOS_DIR, video.edited_filename)
            if not os.path.exists(edited_video_path):
                raise FileNotFoundError(f"Edited video file not found: {edited_video_path}")
            start_time = short.start_time
            end_time = short.end_time
            
            # Calculate duration in seconds
            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)
            duration = end_seconds - start_seconds
            
            # New filename format: shortname-short_description.mp4
            short_filename = f"{short.short_name}-{short.short_description.replace(' ', '_').replace('/', '-')}.mp4"
            short_filename = secure_filename(short_filename)  # Sanitize the filename
            short_path = os.path.join(EDITED_SHORTS_DIR, short_filename)
            
            subprocess.run([
                "ffmpeg", "-ss", start_time, "-i", edited_video_path, "-t", str(duration),
                "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", short_path
            ], check=True)
            
            short.short_filename = short_filename
            short.status = 'completed'
            db.session.commit()
            logger.info(f"Short {short_id} created successfully from edited video.")
        except Exception as e:
            short.status = 'failed'
            logger.error(f"Error creating short {short_id}: {e}")
            db.session.commit()

def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Endpoints
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(VIDEOS_DIR, filename)
    video_file.save(video_path)

    video = Video(original_filename=os.path.basename(video_path), status='pending')
    db.session.add(video)
    db.session.commit()

    threading.Thread(target=process_uploaded_video, args=(video.id,)).start()
    return jsonify({'video_id': video.id}), 202

@app.route('/videos', methods=['GET'])
def get_videos():
    videos = Video.query.filter_by(status='completed').all()
    return jsonify([{
        'id': v.id,
        'title': v.video_title,
        'edited_video_url': url_for('serve_edited_video', filename=v.edited_filename, _external=True) if v.edited_filename else None,
        'shorts': [{
            'id': s.id,
            'short_name': s.short_name,
            'short_description': s.short_description,
            'start_time': s.start_time,
            'end_time': s.end_time,
            'status': s.status,
            'short_url': url_for('serve_short', filename=s.short_filename, _external=True) if s.short_filename else None
        } for s in ShortSegment.query.filter_by(video_id=v.id).all()]
    } for v in videos])

@app.route('/videos/<int:video_id>/shorts/<int:short_id>/create', methods=['POST'])
def create_short(video_id, short_id):
    short = ShortSegment.query.get(short_id)
    if not short or short.video_id != video_id:
        return jsonify({'error': 'Short not found'}), 404
    if short.status != 'pending':
        return jsonify({'error': 'Short already processed'}), 400
    threading.Thread(target=process_short, args=(video_id, short_id)).start()
    return jsonify({'message': 'Short creation started'}), 202

@app.route('/videos/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    try:
        # Delete files
        for path, filename in [
            (VIDEOS_DIR, video.original_filename),
            (EDITED_VIDEOS_DIR, video.edited_filename)
        ]:
            file_path = os.path.join(path, filename)
            if filename and os.path.exists(file_path):
                os.remove(file_path)
        for short in ShortSegment.query.filter_by(video_id=video_id).all():
            if short.short_filename:
                short_path = os.path.join(EDITED_SHORTS_DIR, short.short_filename)
                if os.path.exists(short_path):
                    os.remove(short_path)
            db.session.delete(short)
        db.session.delete(video)
        db.session.commit()
        return jsonify({'message': 'Video deleted'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
# In app.py, add this new function above the endpoints
def refresh_video_shorts(video_id):
    with app.app_context():
        video = Video.query.get(video_id)
        if not video:
            return
        try:
            video.status = 'processing'
            db.session.commit()

            # Use existing transcript if available, otherwise re-transcribe
            if video.transcript:
                logger.info(f"Reusing existing transcript for video {video_id}.")
                formatted_transcript = video.transcript
            else:
                logger.info(f"No existing transcript found for video {video_id}, re-transcribing.")
                original_video_path = os.path.join(VIDEOS_DIR, video.original_filename)
                transcript = process_video(original_video_path, video.edited_filename, skip_editing=True)
                formatted_transcript = format_transcript(transcript)
                video.transcript = formatted_transcript  # Save it for future use
                db.session.commit()

            # Get new segments from Gemini
            response_text = get_suggested_segments(formatted_transcript)
            new_segments = parse_segments(response_text)

            # Keep completed shorts and remove others
            existing_shorts = ShortSegment.query.filter_by(video_id=video_id).all()
            completed_shorts = {s.id: s for s in existing_shorts if s.status == 'completed'}
            for short in existing_shorts:
                if short.status != 'completed':
                    if short.short_filename:
                        short_path = os.path.join(EDITED_SHORTS_DIR, short.short_filename)
                        if os.path.exists(short_path):
                            os.remove(short_path)
                    db.session.delete(short)

            # Add new segments, avoiding duplicates with completed shorts
            existing_short_names = {s.short_name for s in completed_shorts.values()}
            for seg in new_segments:
                if seg['short_name'] not in existing_short_names:
                    segment = ShortSegment(
                        video_id=video_id,
                        short_name=seg['short_name'],
                        short_description=seg['short_description'],
                        start_time=seg['start_time'],
                        end_time=seg['end_time'],
                        status='pending'
                    )
                    db.session.add(segment)

            video.status = 'completed'
            db.session.commit()
            logger.info(f"Shorts refreshed for video {video_id}.")
        except Exception as e:
            video.status = 'failed'
            logger.error(f"Error refreshing shorts for video {video_id}: {e}", exc_info=True)
            db.session.commit()

@app.route('/videos/<int:video_id>/shorts/<int:short_id>/update', methods=['POST'])
def update_short(video_id, short_id):
    short = ShortSegment.query.get(short_id)
    if not short or short.video_id != video_id:
        return jsonify({'error': 'Short not found'}), 404
    
    data = request.get_json()
    if not data or 'start_time' not in data or 'end_time' not in data:
        return jsonify({'error': 'Missing start_time or end_time'}), 400
    
    start_time = data['start_time']
    end_time = data['end_time']
    
    # Optional: Add validation for time format (hh:mm:ss)
    import re
    time_pattern = r'^\d{2}:\d{2}:\d{2}$'
    if not (re.match(time_pattern, start_time) and re.match(time_pattern, end_time)):
        return jsonify({'error': 'Invalid time format. Use hh:mm:ss'}), 400
    
    try:
        short.start_time = start_time
        short.end_time = end_time
        db.session.commit()
        return jsonify({'message': 'Short updated successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating short {short_id}: {e}")
        return jsonify({'error': 'Failed to update short'}), 500

# Add the new endpoint below the existing endpoints
@app.route('/videos/<int:video_id>/refresh', methods=['POST'])
def refresh_shorts(video_id):
    video = Video.query.get(video_id)
    if not video:
        return jsonify({'error': 'Video not found'}), 404
    if video.status == 'processing':
        return jsonify({'error': 'Video is already being processed'}), 400
    threading.Thread(target=refresh_video_shorts, args=(video_id,)).start()
    return jsonify({'message': 'Shorts refresh started'}), 202

@app.route('/videos/<int:video_id>/shorts/<int:short_id>/recreate', methods=['POST'])
def recreate_short(video_id, short_id):
    short = ShortSegment.query.get(short_id)
    if not short or short.video_id != video_id:
        return jsonify({'error': 'Short not found'}), 404
    if short.status != 'completed':
        return jsonify({'error': 'Short is not completed'}), 400
    # Reset status and remove existing file
    short.status = 'pending'
    if short.short_filename:
        short_path = os.path.join(EDITED_SHORTS_DIR, short.short_filename)
        if os.path.exists(short_path):
            os.remove(short_path)
        short.short_filename = None
    db.session.commit()
    threading.Thread(target=process_short, args=(video_id, short_id)).start()
    return jsonify({'message': 'Short recreation started'}), 202

@app.route('/edited-videos/<path:filename>')
def serve_edited_video(filename):
    return send_from_directory(EDITED_VIDEOS_DIR, filename)

@app.route('/shorts/<path:filename>')
def serve_short(filename):
    return send_from_directory(EDITED_SHORTS_DIR, filename)

@app.route('/')
def index():
    return app.send_static_file('index.html')

import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external address to get the local IP (doesn’t send data)
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'  # Fallback to localhost if detection fails
    finally:  
        s.close()
    return ip

if __name__ == '__main__':
    local_ip = get_local_ip()
    print(f"Server running at http://{local_ip}:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)