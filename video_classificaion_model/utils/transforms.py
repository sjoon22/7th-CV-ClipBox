import torchvision.transforms as T

def get_transforms(mode='train', image_size=224):
    """
    이미지 전처리용 transform 생성 함수

    Args:
        mode (str): 'train' or 'val' or 'test'
        image_size (int): 이미지 리사이즈 크기 (224 등)

    Returns:
        torchvision.transforms.Compose 객체
    """
    if mode == 'train':
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            # 영상 classification은 temporal consistency 중요 → augmentation 최소화
            # T.RandomHorizontalFlip(),  # 사용할 경우 LSTM 일관성 깨질 수 있음
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])
    else:  # 'val' or 'test'
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])
    return transform
