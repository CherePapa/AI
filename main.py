import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN
from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser(
        description="Скрипт для обучения или тестирования модели нейронной сети на изображениях."
    )
    parser.add_argument(
        'mode',
        choices=['train', 'test'],
        help="Режим запуска скрипта: 'train' для обучения модели, 'test' для тестирования модели."
    )

    args = parser.parse_args()

    # Определение преобразований для данных
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Загрузка данных
    if args.mode == 'train':
        dataset = datasets.ImageFolder(root='./mnist/training', transform=transform)
    else:
        dataset = datasets.ImageFolder(root='./mnist/testing', transform=transform)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=(args.mode == 'train'))

    # Инициализация модели
    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes=num_classes)

    if args.mode == 'train':
        # Обучение модели
        criterion = torch.nn.CrossEntropyLoss() #функция измерение разлиичий между предсказанами ии и фактически целивыми значениями
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #вариант оптимизации для повышения оптимизации ии
        train_model(model, data_loader, criterion, optimizer)
    else:
        # Тестирование модели
        model.load_state_dict(torch.load('model.pth'))
        model.eval()
        test_model(model, data_loader)

if __name__ == "__main__":
    import sys
    main()
