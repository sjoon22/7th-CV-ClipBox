#영상재추출
import os
import cv2
import natsort
from tqdm import tqdm
from pathlib import Path

def create_video_reviewed(input_folder: str, output_video_path: str, fps: float) -> None:
    """
    지정된 폴더의 이미지 프레임들을 모아 하나의 영상 파일로 생성합니다. (개선안 적용)

    Args:
        input_folder (str): 이미지 프레임들이 있는 폴더 경로.
        output_video_path (str): 생성될 영상 파일의 경로 (예: './output.mp4').
        fps (float): 영상의 초당 프레임 수 (FPS). 원본 영상과 맞추는 것이 좋습니다.
    """
    print(f"\n'{input_folder}'의 프레임들로 영상 생성을 시작합니다...")
    
    input_path = Path(input_folder)
    
    # pathlib을 사용하여 이미지 파일 목록을 안전하게 가져옵니다.
    image_files = natsort.natsorted(
        [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    )

    if not image_files:
        print(f"오류: '{input_folder}'에 이미지 파일이 없습니다.")
        return

    # 첫 번째 이미지를 안전하게 읽어 영상 크기를 결정합니다.
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"오류: 첫 번째 프레임 '{image_files[0].name}'을 읽을 수 없습니다.")
        return
        
    height, width, _ = first_frame.shape
    frame_size = (width, height)

    # 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # tqdm을 사용하여 진행 상황을 시각적으로 표시합니다.
    for image_path in tqdm(image_files, desc="영상 생성 중"):
        frame = cv2.imread(str(image_path))
        if frame is not None:
            # 혹시 모를 프레임 크기 불일치에 대비해 리사이즈 (더 높은 안정성)
            if (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size)
            video_writer.write(frame)
        else:
            print(f"\n경고: '{image_path.name}' 프레임을 읽을 수 없어 건너뜁니다.")

    video_writer.release()
    print(f"\n영상 생성 완료! 저장 경로: {output_video_path}")



