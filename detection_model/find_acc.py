import os
import glob
import random
import shutil
import classification
from classification import *

def find_first_accident(frames_folder_path, window_size=100, step_size=20, threshold=0.8):
    """
    이미지 프레임 폴더를 분석하여 '최초의' 사고 발생 구간을 탐지하고 즉시 중단합니다.

    Args:
        frames_folder_path (str): 프레임 이미지들이 저장된 폴더 경로
        window_size (int): 한 번에 분석할 프레임의 수 (구간 크기)
        step_size (int): 다음 구간으로 이동할 때 건너뛸 프레임의 수
        threshold (float): 사고로 판단할 모델의 출력 임계값

    Returns:
        tuple: 최초로 탐지된 사고 구간의 (시작 인덱스, 끝 인덱스). 없으면 None.
    """
    try:
        image_paths = glob.glob(os.path.join(frames_folder_path, '*.[jp][pn]g'))
        all_frames = sorted(image_paths)
    except FileNotFoundError:
        print(f"오류: 폴더를 찾을 수 없습니다 - {frames_folder_path}")
        return None

    if not all_frames:
        print(f"오류: 폴더에 이미지 파일이 없습니다 - {frames_folder_path}")
        return None

    num_frames = len(all_frames)
    print(f"총 {num_frames}개의 프레임을 찾았습니다. 분석을 시작합니다.")

    for start_index in range(0, num_frames - window_size + 1, step_size):
        end_index = start_index + window_size
        current_chunk = all_frames[start_index:end_index]
        
        # current_chunk가 있는 새로운 폴더 생성
        chunk_folder_name = f"chunk_{start_index}_to_{end_index-1}"
        current_chunk_path = os.path.join('./chunks/', chunk_folder_name)
        os.makedirs(current_chunk_path, exist_ok=True)

        # 이미지 이동
        for img_path in current_chunk:
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(current_chunk_path, img_name)
            shutil.copy(img_path, dst_path)

        print(f"\n구간 [{start_index}, {end_index}] 분석 중...")
        model_output = check_accident(current_chunk_path)
        print(f"  - 모델 출력값: {model_output:.4f}")

        # 임계값을 넘으면 즉시 해당 구간을 반환하고 함수를 완전히 종료
        if model_output > threshold:
            print(f"  => 사고 탐지! (임계값 {threshold} 초과)")
            return current_chunk_path

    # for 루프가 모두 끝날 때까지 사고가 탐지되지 않은 경우
    return None