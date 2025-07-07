from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify,session,flash
import os
import time
from werkzeug.utils import secure_filename

import sys
from video2frames import *
from find_acc import *
from detection import *
from make_video import *
import cv2


from pymongo import MongoClient
import requests
client = MongoClient("mongodb://localhost:27017/")
db = client.clipbox_db
users_collection = db.users
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

KAKAO_USER_INFO_URL = "https://kapi.kakao.com/v2/user/me"
app = Flask(__name__,static_folder=STATIC_DIR)
app.secret_key = 'clipbox-secret-key' 

UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(STATIC_DIR, 'processed')
TEMP_FRAMES_FOLDER = os.path.join(STATIC_DIR, 'temp', 'frames')
TEMP_BLURRED_FOLDER = os.path.join(STATIC_DIR, 'temp', 'blurred')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# 아래 두 줄 추가
os.makedirs(TEMP_FRAMES_FOLDER, exist_ok=True)
os.makedirs(TEMP_BLURRED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
@app.route('/logout')
def logout():
    session.pop('nickname', None)
    return redirect(url_for('home'))

def make_blurred_video(FRAME_DIR, YOLO_MODEL_PATH, BLURRED_OUTPUT_DIR, filename):
    src = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dst = os.path.join(app.config['PROCESSED_FOLDER'], f'blurred_{filename}')
    
    try:
        processor = DetectionProcessor(frames_folder_path=FRAME_DIR, yolo_model_path=YOLO_MODEL_PATH)
        processor.process_video()
        processor.apply_blur_to_video(output_folder=BLURRED_OUTPUT_DIR)
        print("\n" + "="*40 + "\n결과 비교: 프레임 0\n" + "="*40)
        processor.visualize_frame(frame_index=5)
        processor.visualize_blurred_frame(frame_index=5)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")

    blurred_frames_folder = BLURRED_OUTPUT_DIR
    fps = 30
    temp_video_path = os.path.join(PROCESSED_FOLDER, f'temp_{filename}')
    final_video_path = os.path.join(PROCESSED_FOLDER, f'blurred_{filename}')

    create_video_reviewed(blurred_frames_folder, temp_video_path, fps)
    reencode_video_with_ffmpeg(temp_video_path, final_video_path)

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

    # 썸네일 생성
    cap = cv2.VideoCapture(src)
    success, frame = cap.read()
    if success:
        thumb_path = os.path.join("static/images", "recent.jpg")
        cv2.imwrite(thumb_path, frame)
    cap.release()

    return f'blurred_{filename}'


uploaded_list = []
uploaded_with_location = []
import subprocess

def reencode_video_with_ffmpeg(input_path, output_path):
    """
    ffmpeg_path = "C:/vscode_chaniii/ffmpeg-2025-07-01-git-11d1b71c31-essentials_build/ffmpeg-2025-07-01-git-11d1b71c31-essentials_build/ffmpeg-2025-07-01-git-11d1b71c31-essentials_build/bin/ffmpeg.exe"
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-strict", "-2",
        output_path
    ]
    subprocess.run(cmd, check=True)


@app.route('/list')
def show_list():
    if 'nickname' not in session:
        return redirect(url_for('login'))

    user = users_collection.find_one({"nickname": session['nickname']})
    if not user:
        return redirect(url_for('login'))

    user_videos = list(db.uploaded_videos.find({"kakao_id": user["kakao_id"]}, {'_id': 0}).sort("timestamp", -1))
    return render_template('list.html', videos=user_videos)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/kakao-login', methods=['POST'])
def kakao_login():
    data = request.get_json()
    token = data.get('token')

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"
    }

    response = requests.post(KAKAO_USER_INFO_URL, headers=headers)
    if response.status_code != 200:
        return jsonify({"error": "카카오 사용자 정보 요청 실패"}), 400

    user_info = response.json()
    kakao_id = user_info.get('id')
    nickname = user_info.get('properties', {}).get('nickname', '사용자')
    profile_image = user_info.get('properties', {}).get('profile_image')

    # MongoDB 저장 (중복 확인)
    if not users_collection.find_one({"kakao_id": kakao_id}):
        users_collection.insert_one({
            "kakao_id": kakao_id,
            "nickname": nickname,
            "profile_image": profile_image
        })

    # 세션에 저장
    session['nickname'] = nickname
    session['profile_image'] = profile_image

    return jsonify({"nickname": nickname})


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload-video', methods=['POST'])
def upload_video():
    file = request.files['video']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    input_video = f'{UPLOAD_FOLDER}/{filename}'
    extract_frames(video_file=input_video, save_dir=TEMP_FRAMES_FOLDER) 

    FRAME_DIR = find_first_accident(TEMP_FRAMES_FOLDER)
    if FRAME_DIR:
        return jsonify({'result': 'accident', 'filename': filename, 'FRAME_DIR': FRAME_DIR})
    else:
        return jsonify({'result': 'normal'})
@app.route('/extract', methods=['POST'])
def extract():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid or missing JSON'}), 400

    filename = data.get('filename')
    FRAME_DIR = data.get('FRAME_DIR')

    if not filename or not FRAME_DIR:
        return jsonify({'error': 'Missing filename or FRAME_DIR'}), 400

    YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'best_1.pt') 
    """
    BLURRED_OUTPUT_DIR = "C:/vscode_chaniii/integrated_model/blurred_result"
    """
    processed = make_blurred_video(FRAME_DIR, YOLO_MODEL_PATH, TEMP_BLURRED_FOLDER, filename)
    return jsonify({'processed_filename': processed})

@app.route('/upload-to-list', methods=['POST'])
def upload_to_list():
    if 'nickname' not in session or 'profile_image' not in session:
        return jsonify({'error': '로그인이 필요합니다'}), 403

    kakao_id = None
    user = users_collection.find_one({"nickname": session['nickname']})
    if user:
        kakao_id = user.get("kakao_id")

    if not kakao_id:
        return jsonify({'error': '사용자 정보를 찾을 수 없습니다'}), 404

    data = request.json
    processed_filename = data.get('filename')
    lat = data.get('lat', 37.5665)
    lng = data.get('lng', 126.9780)

    # 업로드 DB에 저장
    db.uploaded_videos.insert_one({
        'kakao_id': kakao_id,
        'filename': processed_filename,
        'lat': lat,
        'lng': lng,
        'timestamp': time.time()
    })

    return jsonify({'status': 'done'})


reports_collection = db.reports
@app.route('/report')
def report():
    return render_template('report.html')
@app.route('/submit-report', methods=['POST'])
def submit_report():
    title = request.form.get('title')
    description = request.form.get('description')
    video = request.form.get('video') or None

    report_data = {
        'title': title,
        'description': description,
        'video': video,
        'timestamp': datetime.now()
    }

    if 'nickname' in session:
        report_data['nickname'] = session['nickname']
    if 'kakao_id' in session:
        report_data['kakao_id'] = session['kakao_id']

    reports_collection.insert_one(report_data)
    flash("제보가 성공적으로 접수되었습니다.")
    return redirect(url_for('report'))

@app.route('/map')
def show_map():
    all_videos = list(db.uploaded_videos.find({}, {'_id': 0}))
    return render_template('map.html', locations=all_videos)

from datetime import datetime
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, datetime):
        return value.strftime(format)
    try:
        return datetime.fromisoformat(str(value)).strftime(format)
    except Exception:
        return str(value)
    
@app.route('/recent')
def recent():
    # 모든 사용자의 비디오를 조회
    all_videos = list(db.uploaded_videos.find({}, {'_id': 0}).sort("timestamp", -1))
    return render_template('list.html', videos=all_videos)


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)