import torch
import matplotlib.pyplot as plt

def test_model(model, test_loader, num_iterations=3, num_examples=5):
    accuracies = []
    for i in range(num_iterations):
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"Iteration {i+1}/{num_iterations} - Accuracy: {accuracy:.2f}%")

        # Вывод примеров
        print(f"\nExamples from iteration {i+1}:")
        for j in range(num_examples):
            idx = j  # Вы можете изменить логику выбора примеров
            print(f"  Image {j+1}: Predicted = {all_preds[idx]}, True = {all_labels[idx]}")

    average_accuracy = sum(accuracies) / num_iterations
    print(f"\nAverage Accuracy over {num_iterations} iterations: {average_accuracy:.2f}%")

    # Визуализация примеров
    inputs, labels = next(iter(test_loader))
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    plt.figure(figsize=(15, 5))
    for j in range(num_examples):
        plt.subplot(2, (num_examples + 1) // 2, j + 1)  # Гибкая сетка
        plt.imshow(inputs[j].permute(1, 2, 0).cpu().numpy())
        plt.title(f"Pred: {predicted[j]}\nTrue: {labels[j]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
