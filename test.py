import torch
import matplotlib.pyplot as plt

def test_model(model, test_loader, num_iterations=3, num_examples=5):
    """
    Тестирование модели на тестовых данных с визуализацией результатов
    
    Параметры:
        model: обученная модель для тестирования
        test_loader: DataLoader для тестового набора данных
        num_iterations: количество прогонов на тестовом наборе (по умолчанию 3)
        num_examples: количество примеров для визуализации (по умолчанию 5)
    """
    accuracies = []  # Список для хранения точности по итерациям
    
    # Цикл тестирования на нескольких итерациях
    for i in range(num_iterations):
        correct = 0   # Счетчик правильных предсказаний
        total = 0     # Общее количество примеров
        all_preds = []  # Сохранение всех предсказаний
        all_labels = [] # Сохранение всех истинных меток
        
        # Отключаем вычисление градиентов для ускорения работы
        with torch.no_grad():
            # Итерация по батчам тестовых данных
            for inputs, labels in test_loader:
                # Прямой проход: получение предсказаний модели
                outputs = model(inputs)
                
                # Получение предсказанных классов (индексы с максимальной вероятностью)
                _, predicted = torch.max(outputs.data, 1)
                
                # Обновление счетчиков
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Сохранение результатов для анализа
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Расчет точности для текущей итерации
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
        # Вывод статистики
        print(f"Iteration {i+1}/{num_iterations} - Accuracy: {accuracy:.2f}%")
        
        # Вывод примеров предсказаний из текущей итерации
        print(f"\nExamples from iteration {i+1}:")
        for j in range(num_examples):
            idx = j  # Берем первые N примеров (можно изменить логику выбора)
            print(f"  Image {j+1}: Predicted = {all_preds[idx]}, True = {all_labels[idx]}")
    
    # Расчет и вывод средней точности
    average_accuracy = sum(accuracies) / num_iterations
    print(f"\nAverage Accuracy over {num_iterations} iterations: {average_accuracy:.2f}%")
    
    # Визуализация примеров с предсказаниями
    # Получаем один батч данных для визуализации
    inputs, labels = next(iter(test_loader))
    
    # Делаем предсказания для этого батча
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    
    # Настройка области для отрисовки
    plt.figure(figsize=(15, 5))
    
    # Отрисовка примеров с предсказаниями
    for j in range(num_examples):
        plt.subplot(2, (num_examples + 1) // 2, j + 1)  # Динамическая сетка (2 строки)
        
        # Трансформация тензора изображения для matplotlib 
        # (меняем порядок каналов: CxHxW -> HxWxC)
        img = inputs[j].permute(1, 2, 0).cpu().numpy()
        
        plt.imshow(img)
        plt.title(f"Pred: {predicted[j]}\nTrue: {labels[j]}")  # Заголовок с предсказанием и истинной меткой
        plt.axis('off')  # Скрываем оси
    
    plt.tight_layout()  # Автоматическая настройка отступов
    plt.show()