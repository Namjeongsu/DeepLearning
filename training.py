import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from common.trainer import Trainer


def load_images_from_folder(folder_path, target_size=(28, 28), normalize=True):
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):  # .jpg 파일만 처리
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')  # 이미지를 흑백으로 변환
            img = img.resize(target_size)  # 이미지 크기 조정
            img = np.array(img)  # 이미지를 numpy 배열로 변환

            if normalize:  # 정규화 여부 체크
                img = img / 255.0  # 0~1 정규화

            img = img.reshape(1, target_size[0], target_size[1])  # 채널 추가
            images.append(img)

            # 레이블을 파일명에서 가져오기
            try:
                label = int(filename.split('_')[1].split('.')[0])  # 'image_0.jpg'에서 '0'을 가져옴
                labels.append(label)
                print(f"Extracted label: {label} from {filename}")  # 레이블 추출 디버깅
            except (IndexError, ValueError):
                print(f"Warning: Unable to extract label from filename '{filename}'. Ignoring this file.")

    return np.array(images), np.array(labels)


def _change_one_hot_label(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes), dtype=int)

    for idx, label in enumerate(labels):
        if 0 <= label < num_classes:  # 유효한 레이블만 처리
            one_hot[idx, label] = 1
        else:
            print(f"Warning: label {label} out of bounds for index {idx}.")  # 잘못된 레이블 경고

    return one_hot


if __name__ == '__main__':
    # 데이터 읽기
    x_train, t_train = load_images_from_folder('Images', normalize=True)
    x_test, t_test = load_images_from_folder('TestSet', normalize=True)

    # 최대 레이블 값으로 num_classes 설정
    num_classes = max(max(t_train), max(t_test)) + 1

    # 레이블을 원-핫 인코딩으로 변환
    t_train = _change_one_hot_label(t_train, num_classes)
    t_test = _change_one_hot_label(t_test, num_classes)

    # 데이터 크기 확인
    print(f"x_train shape: {x_train.shape}, t_train shape: {t_train.shape}")
    print(f"x_test shape: {x_test.shape}, t_test shape: {t_test.shape}")

    max_epochs = 20

    # 네트워크 초기화
    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=num_classes, weight_init_std=0.01)

    # 배치 사이즈 조정
    mini_batch_size = min(len(x_train), 100)  # 데이터 크기와 배치 사이즈 비교
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=mini_batch_size,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # 매개변수 보존
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # 그래프 그리기
    markers = {'train': 'o', 'test': 's'}

    # train_acc_list와 test_acc_list의 길이 확인
    print(f"Train Accuracy List Length: {len(trainer.train_acc_list)}")
    print(f"Test Accuracy List Length: {len(trainer.test_acc_list)}")

    x = np.arange(len(trainer.train_acc_list))  # train_acc_list의 길이에 맞추어 x 생성
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
