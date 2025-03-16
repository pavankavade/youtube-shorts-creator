#app.py
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
import threading
import os
import json
from createshorts import main, EDITED_VIDEOS_DIR, EDITED_SHORTS_DIR  # Import your modified script

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Task model
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    youtube_url = db.Column(db.String(255), nullable=False)
    shorts_duration = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='pending')
    video_id = db.Column(db.String(50))
    edited_video_filename = db.Column(db.String(255))
    shorts_filenames = db.Column(db.Text)  # JSON string of filenames

# Create database tables
with app.app_context():
    db.create_all()

def process_video(task_id):
    with app.app_context():
        task = Task.query.get(task_id)
        task.status = 'processing'
        db.session.commit()
        try:
            video_id, edited_video_path, shorts_paths = main(task.youtube_url, shorts_duration=task.shorts_duration)
            task.video_id = video_id
            task.edited_video_filename = os.path.basename(edited_video_path)
            task.shorts_filenames = json.dumps([os.path.basename(p) for p in shorts_paths])
            task.status = 'completed'
        except Exception as e:
            task.status = 'failed'
            print(f"Error processing task {task_id}: {e}")
        finally:
            db.session.commit()

@app.route('/tasks', methods=['POST'])
def create_task():
    data = request.json
    youtube_url = data.get('youtube_url')
    shorts_duration = data.get('shorts_duration', 52)
    if not youtube_url:
        return jsonify({'error': 'YouTube URL is required'}), 400
    task = Task(youtube_url=youtube_url, shorts_duration=shorts_duration, status='pending')
    db.session.add(task)
    db.session.commit()
    threading.Thread(target=process_video, args=(task.id,)).start()
    return jsonify({'task_id': task.id}), 202

@app.route('/tasks/<int:task_id>', methods=['GET'])
def get_task_status(task_id):
    task = Task.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    response = {'status': task.status}
    if task.status == 'completed':
        response['edited_video_url'] = url_for('serve_edited_video', filename=task.edited_video_filename, _external=True)
        response['shorts_urls'] = [
            url_for('serve_short', filename=f, _external=True)
            for f in json.loads(task.shorts_filenames or '[]')
        ]
    return jsonify(response)

@app.route('/edited-videos/<path:filename>')
def serve_edited_video(filename):
    return send_from_directory(EDITED_VIDEOS_DIR, filename)

@app.route('/shorts/<path:filename>')
def serve_short(filename):
    return send_from_directory(EDITED_SHORTS_DIR, filename)

@app.route('/')
def index():
    return app.send_static_file('index.html')  # For simplicity, serve as static file

if __name__ == '__main__':
    app.run(debug=True)