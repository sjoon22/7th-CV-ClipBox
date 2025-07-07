from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import natsort
from ultralytics import YOLO

class ImageSequenceDataset(Dataset):
    """
    이미지 시퀀스 폴더를 하나의 데이터 단위로 처리하는 커스텀 데이터셋.

    Args:
        root_dir (str): 각 시퀀스 폴더들을 포함하는 루트 디렉토리 경로.
        transform (callable, optional): 샘플에 적용될 transform.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 루트 디렉토리 내의 하위 디렉토리(각각이 하나의 시퀀스) 리스트를 가져옵니다.
        self.sequence_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        """데이터셋에 포함된 총 시퀀스의 수를 반환합니다."""
        return len(self.sequence_folders)

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)에 해당하는 하나의 이미지 시퀀스 전체를 불러옵니다.

        Returns:
            torch.Tensor: (S, C, H, W) 형태의 텐서.
                          S: 시퀀스 길이 (프레임 수)
                          C: 채널 수
                          H: 높이
                          W: 너비
        """
        # 인덱스에 해당하는 시퀀스 폴더 경로를 가져옵니다.
        seq_folder_path = os.path.join(self.root_dir, self.sequence_folders[idx])

        # 폴더 내의 모든 이미지 파일 리스트를 가져옵니다.
        # natsort.natsorted를 사용하여 'frame_1.png', 'frame_10.png'와 같은 파일명을 올바르게 정렬합니다.
        image_files = natsort.natsorted([f for f in os.listdir(seq_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

        frames = []
        for img_name in image_files:
            img_path = os.path.join(seq_folder_path, img_name)
            # PIL을 사용하여 이미지 로드 (RGB로 변환하여 채널 수를 통일)
            image = Image.open(img_path).convert("RGB")

            # transform이 지정되어 있다면 적용합니다. (e.g., 텐서로 변환, 리사이즈, 정규화)
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # 프레임 리스트를 하나의 텐서로 쌓습니다. (리스트의 각 텐서가 첫번째 차원으로 쌓임)
        # 결과 텐서의 shape: [Sequence_length, Channels, Height, Width]
        sequence_tensor = torch.stack(frames)

        return sequence_tensor
    
#face_detection & blur + plate detection & blur

class DetectionProcessor:

    def __init__(self, frames_folder_path: str, yolo_model_path: str):
        """
        Args:
            frames_folder_path (str): 처리할 영상의 프레임들이 담긴 폴더 경로.
            yolo_model_path (str): 학습된 YOLOv8 번호판 탐지 모델(.pt) 파일 경로.
        """
        # --- 프레임 경로 설정 ---
        if not os.path.isdir(frames_folder_path):
            raise FileNotFoundError(f"프레임 폴더를 찾을 수 없습니다: {frames_folder_path}")
        self.frames_folder_path = frames_folder_path
        self.image_paths = natsort.natsorted(
            [os.path.join(self.frames_folder_path, f) for f in os.listdir(self.frames_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )
        if not self.image_paths:
            print(f"경고: '{frames_folder_path}' 폴더에 이미지 파일이 없습니다.")

        # --- YOLOv8 모델 로드 ---
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print("YOLOv8 모델 로드 성공.")
        except Exception as e:
            raise FileNotFoundError(f"YOLOv8 모델 로드 실패: {e}\n모델 경로({yolo_model_path})가 올바른지 확인하세요.")

        # --- 결과 저장 구조 ---
        # detection_results: [{'frame_path': '...', 'detections': [{'box': (l,t,r,b), 'label': 'face'}, ...]}, ...]
        self.detection_results = []
        self.blurred_image_paths = []
        print(f"'{os.path.basename(frames_folder_path)}' 폴더에서 {len(self.image_paths)}개의 프레임을 로드했습니다.")

    def _detect_faces(self, image_np: np.ndarray) -> list:
        """한 이미지에서 얼굴을 탐지하고 표준화된 BBox 리스트를 반환."""
        face_locations = face_recognition.face_locations(image_np, model="hog")
        # (top, right, bottom, left) -> (left, top, right, bottom)으로 변환
        return [(left, top, right, bottom) for top, right, bottom, left in face_locations]

    def _detect_plates(self, image_path: str) -> list:
        """한 이미지에서 번호판을 탐지하고 표준화된 BBox 리스트를 반환."""
        results = self.yolo_model.predict(source=image_path, verbose=False) # predict 로그 끔
        plate_locations = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int) # [x1, y1, x2, y2]
                plate_locations.append(tuple(xyxy)) # (left, top, right, bottom)
        return plate_locations

    def process_video(self):
        """전체 비디오 프레임에 대해 얼굴과 번호판 탐지를 모두 수행하고 결과를 통합."""
        print("\n탐지 프로세스를 시작합니다 (얼굴 & 번호판)...")
        if self.detection_results:
            print("이미 탐지가 완료되었습니다.")
            return

        for i, frame_path in enumerate(self.image_paths):
            print(f"  - 처리 중: 프레임 {i + 1}/{len(self.image_paths)}", end='\r')
            image_np = np.array(Image.open(frame_path).convert("RGB"))

            all_detections = []

            # 1. 얼굴 탐지 및 결과 추가
            face_boxes = self._detect_faces(image_np)
            for box in face_boxes:
                all_detections.append({'box': box, 'label': 'face'})

            # 2. 번호판 탐지 및 결과 추가
            plate_boxes = self._detect_plates(frame_path)
            for box in plate_boxes:
                all_detections.append({'box': box, 'label': 'plate'})

            # 3. 통합된 결과 저장
            self.detection_results.append({
                'frame_path': frame_path,
                'detections': all_detections
            })
        print("\n모든 프레임의 탐지 및 통합 완료.")

    def apply_blur_to_video(self, output_folder: str, kernel_size=(99, 99), sigma=30):
        """통합된 탐지 결과에 블러 처리를 적용하고 이미지를 저장합니다."""
        if not self.detection_results:
            print("오류: 먼저 process_video()를 실행해야 합니다.")
            return

        print(f"\n블러 처리를 시작합니다. 결과는 '{output_folder}' 폴더에 저장됩니다.")
        os.makedirs(output_folder, exist_ok=True)
        self.blurred_image_paths = []

        for i, frame_data in enumerate(self.detection_results):
            print(f"  - 블러 처리 중: 프레임 {i + 1}/{len(self.detection_results)}", end='\r')

            image_np = np.array(Image.open(frame_data['frame_path']).convert("RGB"))
            detections = frame_data['detections']

            if not detections:
                blurred_image = image_np
            else:
                blurred_image = np.copy(image_np)
                for det in detections:
                    left, top, right, bottom = det['box']
                    # 얼굴/번호판 영역 블러 처리
                    region_to_blur = blurred_image[top:bottom, left:right]
                    # 영역이 비어있지 않은 경우에만 블러 적용
                    if region_to_blur.size > 0:
                        blurred_region = cv2.GaussianBlur(region_to_blur, kernel_size, sigma)
                        blurred_image[top:bottom, left:right] = blurred_region

            output_path = os.path.join(output_folder, os.path.basename(frame_data['frame_path']))
            cv2.imwrite(output_path, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
            self.blurred_image_paths.append(output_path)
        print("\n블러 처리 완료.")

    def visualize_frame(self, frame_index: int):
        """특정 프레임의 모든 탐지(얼굴, 번호판) 결과를 시각화합니다."""
        if not self.detection_results or not 0 <= frame_index < len(self.detection_results):
            print("오류: process_video()를 먼저 실행하거나 올바른 인덱스를 입력하세요.")
            return

        frame_data = self.detection_results[frame_index]
        print(f"\n--- 원본 프레임 {frame_index} 시각화 (탐지 영역 표시) ---")
        image = Image.open(frame_data['frame_path']).convert("RGB")
        draw = ImageDraw.Draw(image)

        for det in frame_data['detections']:
            box = det['box']
            label = det['label']
            color = "red" if label == 'face' else "blue" # 얼굴:빨강, 번호판:파랑
            draw.rectangle(box, outline=color, width=4)
            draw.text((box[0], box[1] - 10), label, fill=color)

        display(image)

    def visualize_blurred_frame(self, frame_index: int):
        """블러 처리된 특정 프레임을 시각화합니다."""
        if not self.blurred_image_paths or not 0 <= frame_index < len(self.blurred_image_paths):
            print("오류: apply_blur_to_video()를 먼저 실행하거나 올바른 인덱스를 입력하세요.")
            return

        print(f"\n--- 블러 처리된 프레임 {frame_index} 시각화 ---")
        image = Image.open(self.blurred_image_paths[frame_index])
        display(image)
