import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        """
        Простая сверточная нейронная сеть для классификации изображений
        
        Параметры:
            num_classes: количество классов для классификации
        """
        super(SimpleCNN, self).__init__()
        
        # Блок извлечения признаков (feature extraction)
        self.features = nn.Sequential(
            # Первый сверточный слой
            nn.Conv2d(
                in_channels=3,    # 3 канала на входе (RGB)
                out_channels=32,   # 32 фильтра/канала на выходе
                kernel_size=3,     # размер ядра свертки 3x3
                stride=1,          # шаг свертки 1 пиксель
                padding=1          # дополнение нулями по краям
            ),
            nn.ReLU(),            # Активационная функция
            nn.MaxPool2d(
                kernel_size=2,     # размер окна пулинга 2x2
                stride=2           # шаг пулинга 2 пикселя
            ),
            
            # Второй сверточный слой
            nn.Conv2d(
                in_channels=32,    # 32 канала на входе (выход предыдущего слоя)
                out_channels=64,   # 64 фильтра на выходе
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)     # Сокращенная форма записи (kernel_size=2, stride=2)
        )
        
        # Блок классификации (полносвязные слои)
        self.classifier = nn.Sequential(
            # Первый полносвязный слой
            nn.Linear(
                in_features=64 * 32 * 32,  # Размер входа должен соответствовать выходу сверточного блока
                out_features=512          # 512 нейронов на выходе
            ),
            nn.ReLU(),
            
            # Выходной слой
            nn.Linear(
                in_features=512, 
                out_features=num_classes   # Количество классов на выходе
            )
        )

    def forward(self, x):
        """
        Прямой проход данных через сеть
        Аргументы:
            x: входной тензор формы (batch_size, 3, height, width)
        Возвращает:
            Тензор логитов формы (batch_size, num_classes)
        """
        # Проход через сверточные слои
        x = self.features(x)        # Форма на выходе: (batch_size, 64, 32, 32)
        
        # Преобразование 4D тензора в 2D (выравнивание в вектор)
        x = x.view(x.size(0), -1)   # Результат: (batch_size, 64*32*32)
        
        # Проход через полносвязные слои
        x = self.classifier(x)      # Форма на выходе: (batch_size, num_classes)
        
        return x