import torch
from tqdm import tqdm  # Импорт библиотеки для отображения прогресса

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    # Устанавливаем модель в режим обучения (важно для Dropout/BatchNorm)
    model.train()
    
    # Цикл по эпохам
    for epoch in range(num_epochs):
        running_loss = 0.0  # Суммарные потери за эпоху
        
        # Инициализация прогресс-бара для текущей эпохи
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch [{epoch+1}/{num_epochs}]",  # Описание прогресс-бара
            unit="batch"  # Единица измерения - батчи
        )
        
        # Итерация по батчам данных
        for inputs, labels in progress_bar:
            # Обнуление градиентов перед каждым шагом обучения
            optimizer.zero_grad()
            
            # Прямой проход: вычисление выходов модели
            outputs = model(inputs)
            
            # Вычисление значения функции потерь
            loss = criterion(outputs, labels)
            
            # Обратный проход: вычисление градиентов
            loss.backward()
            
            # Обновление весов модели
            optimizer.step()
            
            # Накопление суммарных потерь
            running_loss += loss.item()
            
            # Обновление описания прогресс-бара (средние потери)
            progress_bar.set_postfix(
                loss=running_loss / (progress_bar.n + 1)  # Усреднение по батчам
            )

        # Вывод средних потерь после завершения эпохи
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {epoch_loss:.4f}")

    # Сохранение весов обученной модели
    torch.save(
        model.state_dict(),  # Сохраняем только параметры модели
        'model.pth'          # Имя файла для сохранения
    )
    print("Model weights saved as 'model.pth'")