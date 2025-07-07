# 🚗 7th-CV-ClipBox: AI 사고 영상 임베디드 시스템

---

## 🏷️ 1. 프로젝트 설명

**7th-CV-ClipBox**는  
AI 기반 임베디드 시스템을 활용해  
교통사고 발생 시 자동으로 사고를 감지,  
영상을 수집·분류·공유하는 솔루션입니다.

---

## ⚠️ 2. 문제 상황

<img src="img/문제상황.png" width="66%"/>

블랙박스만으로 사고 입증이 어렵습니다.  
사각지대, 영상 손상, 영상 미존재 등 한계가 있습니다.  
사고 차량 블랙박스만으로는 한계점이 존재합니다.

---

## 🛠️ 3. 해결 방안

<img src="img/해결방안.png" width="66%"/>

AI 임베디드 시스템이 사고를 자동 감지합니다.  
사고 영상은 자동 수집·분류됩니다.  
번호판과 얼굴 등 개인정보는 자동 블러 처리됩니다.  
영상은 실시간으로 업로드·공유됩니다.

---

## 🔄 4. 파이프라인

<img src="img/파이프라인.png" width="66%"/>

사고 영상이 업로드됩니다.  
Task1에서 사고 장면을 추출합니다.  
Task2에서 번호판, 얼굴을 블러 처리합니다.  
웹에 영상이 업로드되고 지도 기반으로 공유됩니다.

---

## 🤖 5. 모델

<img src="img/ResNet18.png" width="33%"/>

Task1 사고 영상 분류에는 ResNet18과 LSTM을 결합한 모델을 사용합니다.  
ResNet18의 block4를 제거하여 feature vector 크기를 256으로 줄였습니다.  
LSTM으로 프레임 순차성을 반영합니다.  
100프레임씩, stride 20 sliding window 방식으로 사고 포함 여부를 분류합니다.  
최초 사고 구간 100프레임과 150프레임을 합쳐 250프레임을 추출합니다.  
Feature vector 256, batch size 16, learning rate 0.0007, dropout 0.5, 3단계마다 0.8배 감소 스케줄러를 사용합니다.

<img src="img/face_recognition_YOLO.png" width="33%"/>

Task2 얼굴 탐지는 Dlib(face_recognition)을 사용합니다.  
HOG와 SVM 기반 이진 분류로 실시간 얼굴 탐지가 가능합니다.

<img src="img/nuScenes.png" width="33%"/>

번호판 탐지는 YOLO(one-stage detection) 모델을 사용합니다.  
실시간으로 바운딩 박스를 반환하며 감지 영역은 자동 블러 처리됩니다.

---

## 🎥 6. 구현 영상

<img src="img/test_video.gif" width="66%"/>

실제 시스템 동작 과정을 시연 영상에서 확인할 수 있습니다.

---
