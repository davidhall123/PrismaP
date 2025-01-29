import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from common import get_session, Book
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import random
import argparse
import os
import numpy as np
import time
import GPUtil

print("Loading genre classes...")
with open('genre_classes.json', 'r') as f:
    genre_classes = json.load(f)

num_classes = len(genre_classes)
print(f"Number of genre classes: {num_classes}")

class BookGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BookGenreClassifier, self).__init__()
        print("Initializing Vision Transformer (ViT) model...")
        self.vit = models.vit_b_16(pretrained=True)
        num_features = self.vit.head.in_features
        self.vit.head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vit(x)
        return self.sigmoid(x)

class BookCoverDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        isbn13, image_path, genre_tensor = self.data[idx]
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
            if self.transform:
                image = self.transform(image)
            return isbn13, image, genre_tensor
        except (IOError, OSError, UnidentifiedImageError):
            print(f"Error loading image: {image_path}")
            return None

def load_data(num_datapoints=None):
    print("Loading data from database...")
    session = get_session()
    query = session.query(Book.isbn13, Book.image_location, Book.encoded_genre).filter(Book.genre != None)
    if num_datapoints:
        query = query.limit(num_datapoints)
    books = query.all()

    data = []
    for book in books:
        if book.isbn13 is not None and book.image_location and os.path.exists(book.image_location) and book.encoded_genre:
            try:
                genre_tensor = torch.tensor(book.encoded_genre)
                data.append((book.isbn13, book.image_location, genre_tensor))
            except (ValueError, TypeError):
                print(f"Error processing genre for ISBN: {book.isbn13}")

    session.close()
    print(f"Loaded {len(data)} datapoints")
    return data

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_gpu_memory_usage():
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
    except:
        return None

def train(model, train_loader, val_loader, num_epochs, learning_rate, device, target_usage=80, accumulation_steps=4):
    print("Starting training...")
    model.to(device)
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            if not batch:  # Skip empty batches
                continue
            _, images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.float())

            # Normalize the loss to account for accumulation steps
            loss = loss / accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            # Check GPU usage and adjust if necessary
            gpu_usage = get_gpu_memory_usage()
            if gpu_usage is not None and gpu_usage > target_usage:
                time.sleep(0.1)  # Small pause to let GPU usage decrease

        # Perform the last optimization step if needed
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if not batch:  # Skip empty batches
                    continue
                _, images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels.float()).item()
        model.train()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"GPU Memory Usage: {get_gpu_memory_usage()} MB")

        if epoch < num_epochs - 1:  # Don't sleep after the last epoch
            print(f"Taking a 60-second break to cool down GPU...")
            time.sleep(60)

def evaluate_model(model, test_loader, device, threshold=0.5, top_n=10, partial_match_threshold=0.5):
    print("Evaluating model...")
    model.eval()
    all_predictions = []
    all_labels = []
    partial_match_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if not batch:  # Skip empty batches
                continue
            _, images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Get top N predictions
            _, top_indices = torch.topk(outputs, top_n)
            predicted = torch.zeros_like(outputs).scatter_(1, top_indices, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate partial match accuracy
            for pred, label in zip(predicted, labels):
                total_samples += 1
                correct_predictions = torch.sum((pred == 1) & (label == 1)).item()
                total_predictions = torch.sum(pred).item()
                if total_predictions > 0 and correct_predictions / total_predictions >= partial_match_threshold:
                    partial_match_correct += 1

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='samples',
                                                               zero_division=0)

    # Calculate Hamming loss
    hamming_loss = np.mean(np.abs(all_predictions - all_labels))

    # Calculate subset accuracy
    subset_accuracy = np.mean(np.all(all_predictions == all_labels, axis=1))

    # Calculate partial match accuracy
    partial_match_accuracy = partial_match_correct / total_samples

    print(f"Sample-averaged Precision: {precision:.4f}")
    print(f"Sample-averaged Recall: {recall:.4f}")
    print(f"Sample-averaged F1-score: {f1:.4f}")
    print(f"Hamming Loss: {hamming_loss:.4f}")
    print(f"Subset Accuracy: {subset_accuracy:.4f}")
    print(f"Partial Match Accuracy: {partial_match_accuracy:.4f}")

    return precision, recall, f1, hamming_loss, subset_accuracy, partial_match_accuracy

def predict_genres(model, isbn13, image_tensor, device, top_n=10):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = outputs[0].cpu().numpy()

        # Get indices of top N predictions
        top_indices = np.argsort(probabilities)[::-1][:top_n]

        # Get genres and their corresponding probabilities
        predicted_genres = [(genre_classes[i], probabilities[i] * 100) for i in top_indices]

        # Sort by probability in descending order
        predicted_genres.sort(key=lambda x: x[1], reverse=True)

    return predicted_genres

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a book genre classifier using Vision Transformer (ViT)")
    parser.add_argument("--num_datapoints", type=int, default=1000000, help="Number of datapoints to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--target_usage", type=int, default=80, help="Target GPU usage percentage")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.num_datapoints} datapoints...")
    data = load_data(args.num_datapoints)

    print("Splitting data into train, validation, and test sets...")
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    print("Creating data loaders...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BookCoverDataset(train_data, transform=transform)
    val_dataset = BookCoverDataset(val_data, transform=transform)
    test_dataset = BookCoverDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    print("Initializing model...")
    model = BookGenreClassifier(num_classes)

    print(f"Training model for {args.num_epochs} epochs with learning rate {args.learning_rate}...")
    train(model, train_loader, val_loader, args.num_epochs, args.learning_rate, device,
          target_usage=args.target_usage, accumulation_steps=args.accumulation_steps)

    print("Evaluating model...")
    precision, recall, f1, hamming_loss, subset_accuracy, partial_match_accuracy = evaluate_model(model, test_loader, device)

    print("Saving trained model...")
    torch.save(model.state_dict(), 'book_genre_classifierV21mil.pth')

    print("Making predictions for 15 random books...")
    random_samples = random.sample(test_data, 15)

    for isbn13, image_path, true_labels in random_samples:
        try:
            with Image.open(image_path) as img:
                image = img.convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            predicted_genres = predict_genres(model, isbn13, image_tensor, device)
            true_genres = [genre_classes[i] for i, label in enumerate(true_labels) if label == 1]

            print(f"\nISBN: {isbn13}")
            print("Predicted genres (with confidence):")
            for genre, confidence in predicted_genres:
                print(f"  {genre}: {confidence:.2f}%")
            print(f"Actual genres: {true_genres}")
        except (IOError, OSError, UnidentifiedImageError):
            print(f"\nError loading image for ISBN: {isbn13}")

    print("Program completed successfully.")
