import torch
import tqdm
from tqdm import tqdm  # Для индикатора прогресса

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Loss: {running_loss/len(train_loader):.4f}")

    # Сохранение обученной модели
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")
