# app.py
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
import threading
import os
import json
from createshorts import main, EDITED_VIDEOS_DIR, EDITED_SHORTS_DIR
import logging

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    youtube_url = db.Column(db.String(255), nullable=False)
    shorts_duration = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='pending')
    video_id = db.Column(db.String(50))
    video_title = db.Column(db.String(512))
    edited_video_filename = db.Column(db.String(512))
    shorts_filenames = db.Column(db.Text)
    transcription_progress = db.Column(db.Float, default=0.0)  # Progress for transcription (0-100%)
    rendering_progress = db.Column(db.Float, default=0.0)      # Progress for video rendering (0-100%)
    shorts_progress = db.Column(db.Float, default=0.0)         # Progress for shorts creation (0-100%)

with app.app_context():
    db.create_all()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_progress(task_id, progress, stage):
    """Update the task's progress and log it."""
    with app.app_context():
        task = Task.query.get(task_id)
        if stage == 'transcribing':
            task.transcription_progress = progress
        elif stage == 'video_processing':
            task.rendering_progress = progress
        elif stage == 'shorts_creation':
            task.shorts_progress = progress
        db.session.commit()
    logger.info(f"Task {task_id} - {stage}: {progress}%")

def process_video(task_id):
    with app.app_context():
        task = Task.query.get(task_id)
        task.status = 'processing'
        db.session.commit()
        try:
            def progress_callback(progress, stage='transcribing'):
                update_progress(task_id, progress, stage)

            video_id, video_title, edited_video_path, shorts_paths = main(
                task.youtube_url,
                task.id,
                shorts_duration=task.shorts_duration,
                update_progress=progress_callback
            )
            task.video_id = video_id
            task.video_title = video_title
            task.edited_video_filename = os.path.basename(edited_video_path)
            task.shorts_filenames = json.dumps([os.path.basename(p) for p in shorts_paths])
            task.status = 'completed'
        except Exception as e:
            task.status = 'failed'
            logger.error(f"Error processing task {task_id}: {e}")
        finally:
            db.session.commit()
            

@app.route('/tasks', methods=['POST'])
def create_task():
    data = request.json
    youtube_url = data.get('youtube_url')
    shorts_duration = data.get('shorts_duration', 52)
    if not youtube_url:
        return jsonify({'error': 'YouTube URL is required'}), 400
    task = Task(youtube_url=youtube_url, shorts_duration=shorts_duration)
    db.session.add(task)
    db.session.commit()
    threading.Thread(target=process_video, args=(task.id,)).start()
    return jsonify({'task_id': task.id}), 202

@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task_status(task_id):
    task = Task.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    response = {
        'status': task.status,
        'progress': {
            'transcription': task.transcription_progress,
            'rendering': task.rendering_progress,
            'shorts': task.shorts_progress
        }
    }
    if task.status == 'completed':
        response['edited_video_url'] = url_for('serve_edited_video', filename=task.edited_video_filename, _external=True)
        response['shorts_urls'] = [
            url_for('serve_short', filename=f, _external=True)
            for f in json.loads(task.shorts_filenames or '[]')
        ]
    return jsonify(response)

@app.route('/tasks', methods=['GET'])
def get_tasks():
    tasks = Task.query.filter_by(status='completed').all()
    tasks_data = [
        {
            'id': task.id,
            'video_title': task.video_title,
            'youtube_url': task.youtube_url,
            'edited_video_url': url_for('serve_edited_video', filename=task.edited_video_filename, _external=True),
            'shorts_urls': [url_for('serve_short', filename=f, _external=True) for f in json.loads(task.shorts_filenames or '[]')]
        }
        for task in tasks
    ]
    return jsonify(tasks_data)

@app.route('/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = Task.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    try:
        # Delete edited video file
        edited_video_path = os.path.join(EDITED_VIDEOS_DIR, task.edited_video_filename)
        if os.path.exists(edited_video_path):
            os.remove(edited_video_path)
        # Delete shorts files
        shorts_filenames = json.loads(task.shorts_filenames or '[]')
        for filename in shorts_filenames:
            short_path = os.path.join(EDITED_SHORTS_DIR, filename)
            if os.path.exists(short_path):
                os.remove(short_path)
        # Delete database record
        db.session.delete(task)
        db.session.commit()
        return jsonify({'message': 'Task deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/edited-videos/<path:filename>')
def serve_edited_video(filename):
    return send_from_directory(EDITED_VIDEOS_DIR, filename)

@app.route('/shorts/<path:filename>')
def serve_short(filename):
    return send_from_directory(EDITED_SHORTS_DIR, filename)

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)