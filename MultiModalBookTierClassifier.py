import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import DebertaTokenizer, DebertaModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import argparse
import os
import numpy as np
from collections import defaultdict
import warnings
from torchvision.transforms import functional as TF
from itertools import cycle
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Import get_session and Book from common module
from common import get_session, Book

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define some constants
MIN_SAMPLES_PER_GENRE = 35000
MAX_SAMPLES_PER_GENRE = 35000
PERCENTILE_TIERS = [20, 40, 60, 80]
num_percentiles = len(PERCENTILE_TIERS) + 1
VALID_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}


class ImageAugmenter:
    """Class to handle image augmentations like flipping, rotating, and adjusting colors."""

    def __init__(self):
        self.augmentation_stats = defaultdict(int)
        self.failed_augmentations = defaultdict(int)

    @staticmethod
    def is_valid_image(image: Image.Image) -> bool:
        """Check if the image meets certain quality criteria."""
        try:
            # Ensure image is large enough
            if image.size[0] < 64 or image.size[1] < 64:
                return False

            # Convert to tensor to check for uniformity
            tensor_image = TF.to_tensor(image)
            if torch.allclose(tensor_image, tensor_image[0, 0, 0]):
                return False

            return True
        except Exception as e:
            # If something goes wrong, consider the image invalid
            return False

    def apply_geometric_augmentation(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """Apply geometric transformations like flips and rotations."""
        augmented_images = []
        try:
            tensor_image = TF.to_tensor(image)

            # Flip the image horizontally
            flipped = TF.hflip(tensor_image)
            pil_flipped = TF.to_pil_image(flipped)
            if self.is_valid_image(pil_flipped):
                augmented_images.append(('flip', pil_flipped))
                self.augmentation_stats['flip'] += 1

            # Rotate the image by various angles
            for angle in [-15, -10, -5, 5, 10, 15]:
                rotated = TF.rotate(tensor_image, angle)
                pil_rotated = TF.to_pil_image(rotated)
                if self.is_valid_image(pil_rotated):
                    augmented_images.append((f'rotate_{angle}', pil_rotated))
                    self.augmentation_stats[f'rotate_{angle}'] += 1

            # Zoom in and out
            for scale in [0.9, 1.1]:
                h, w = tensor_image.shape[-2:]
                new_h = int(h * scale)
                new_w = int(w * scale)
                resized = TF.resize(tensor_image, size=[new_h, new_w])

                if scale < 1:  # Zoom out by padding
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    padded = TF.pad(resized, [pad_h, pad_w], fill=1)
                    if padded.shape[-2:] != (h, w):
                        padded = TF.center_crop(padded, [h, w])
                    zoomed = padded
                else:  # Zoom in by cropping
                    zoomed = TF.center_crop(resized, [h, w])

                pil_zoomed = TF.to_pil_image(zoomed)
                if self.is_valid_image(pil_zoomed):
                    augmented_images.append((f'zoom_{scale}', pil_zoomed))
                    self.augmentation_stats[f'zoom_{scale}'] += 1

        except Exception:
            # If augmentation fails, skip it
            self.failed_augmentations['geometric'] += 1
            return []

        return augmented_images

    def apply_color_augmentation(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """Adjust the brightness, contrast, and saturation of the image."""
        augmented_images = []
        try:
            tensor_image = TF.to_tensor(image)

            # Adjust brightness
            for factor in [0.8, 1.2]:
                brightened = TF.adjust_brightness(tensor_image, factor)
                pil_brightened = TF.to_pil_image(brightened)
                if self.is_valid_image(pil_brightened):
                    augmented_images.append((f'brightness_{factor}', pil_brightened))
                    self.augmentation_stats[f'brightness_{factor}'] += 1

            # Adjust contrast
            for factor in [0.8, 1.2]:
                contrasted = TF.adjust_contrast(tensor_image, factor)
                pil_contrasted = TF.to_pil_image(contrasted)
                if self.is_valid_image(pil_contrasted):
                    augmented_images.append((f'contrast_{factor}', pil_contrasted))
                    self.augmentation_stats[f'contrast_{factor}'] += 1

            # Adjust saturation
            for factor in [0.8, 1.2]:
                saturated = TF.adjust_saturation(tensor_image, factor)
                pil_saturated = TF.to_pil_image(saturated)
                if self.is_valid_image(pil_saturated):
                    augmented_images.append((f'saturation_{factor}', pil_saturated))
                    self.augmentation_stats[f'saturation_{factor}'] += 1

        except Exception:
            # If color adjustment fails, skip it
            self.failed_augmentations['color'] += 1
            return []

        return augmented_images

    def get_augmentation_stats(self) -> Dict[str, int]:
        """Provide a summary of augmentation activities."""
        return {
            'total_augmentations': sum(self.augmentation_stats.values()),
            'failed_augmentations': sum(self.failed_augmentations.values()),
            'augmentation_types': dict(self.augmentation_stats),
            'failure_types': dict(self.failed_augmentations)
        }


def collate_fn(batch):
    """Custom collate function to filter out invalid samples."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    isbn13s, images, genre_indices, percentile_tier_indices, input_ids_list, attention_masks_list = zip(*batch)
    images = torch.stack(images, 0)
    genre_indices = torch.tensor(genre_indices)
    percentile_tier_indices = torch.tensor(percentile_tier_indices)
    input_ids = torch.stack(input_ids_list)
    attention_masks = torch.stack(attention_masks_list)
    return isbn13s, images, genre_indices, percentile_tier_indices, input_ids, attention_masks


def parse_genres(genre_str):
    """Convert genre string to a list of individual genres."""
    if not isinstance(genre_str, str):
        return []

    try:
        genres = json.loads(genre_str)
        if not isinstance(genres, list):
            raise ValueError("Parsed JSON is not a list")
    except (json.JSONDecodeError, ValueError):
        genres = [g.strip().strip('"').strip("'") for g in genre_str.split(',')]

    return [g for g in genres if g]


class BookPercentilePredictor(nn.Module):
    """Neural network model to predict book percentile tiers based on image, genre, and text data."""

    def __init__(self, num_genres: int, num_percentiles: int):
        super(BookPercentilePredictor, self).__init__()

        # Image feature extractor using EfficientNet-B7
        self.efficientnet = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        num_image_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()

        # Additional layers for image features
        self.image_layers = nn.Sequential(
            nn.Linear(num_image_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Genre embedding and processing
        self.genre_embedding = nn.Embedding(num_genres, 384)
        self.genre_layers = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Text feature extractor using DeBERTa
        self.text_model = DebertaModel.from_pretrained('microsoft/deberta-base')
        for param in self.text_model.parameters():
            param.requires_grad = False
        text_embedding_size = self.text_model.config.hidden_size

        # Additional layers for text features
        self.text_layers = nn.Sequential(
            nn.Linear(text_embedding_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Combine all features
        total_features = 512 + 256 + 256
        self.combined_layers = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_percentiles)
        )

    def forward(self, images, genre_indices, input_ids, attention_masks):
        # Extract image features
        image_features = self.efficientnet(images)
        image_features = self.image_layers(image_features)

        # Extract genre features
        genre_features = self.genre_embedding(genre_indices)
        genre_features = self.genre_layers(genre_features)

        # Extract text features
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_masks)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_layers(text_features)

        # Combine all features and make predictions
        combined_features = torch.cat((image_features, genre_features, text_features), dim=1)
        outputs = self.combined_layers(combined_features)
        return outputs


class GenreBalancedDataManager:
    """Class to manage and balance the dataset across different genres."""

    def __init__(self, min_samples: int = MIN_SAMPLES_PER_GENRE,
                 max_samples: int = MAX_SAMPLES_PER_GENRE):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.augmenter = ImageAugmenter()
        self.genre_stats = defaultdict(lambda: {
            'original_count': 0,
            'augmented_count': 0,
            'total_count': 0,
            'percentile_distribution': defaultdict(int),
            'tier_augmentation_counts': defaultdict(int),
            'tier_reduction_counts': defaultdict(int)
        })

    def load_and_balance_data(self) -> Tuple[List, Dict, Dict]:
        """Load data from the database and balance it across genres."""
        # Load books from the database
        session = get_session()
        books = self._load_books_from_db(session)
        session.close()

        # Organize books by genre
        genre_books = self._organize_books_by_genre(books)
        validated_genre_books = self._validate_and_filter_genres(genre_books)

        # Balance the dataset
        balanced_data = self._balance_genres(validated_genre_books)

        # Create a mapping from genre to index
        genre_map = self._create_genre_map(balanced_data)

        # Calculate percentile tiers for each genre
        percentile_tiers = self._calculate_percentile_tiers(balanced_data, genre_map)

        # Prepare the final dataset
        final_dataset = self._prepare_final_dataset(balanced_data, genre_map, percentile_tiers)

        return final_dataset, genre_map, percentile_tiers

    def _load_books_from_db(self, session) -> List[Book]:
        """Fetch books from the database."""
        try:
            query = session.query(
                Book.isbn13,
                Book.image_location,
                Book.genre,
                Book.normalized_ratings_per_year,
                Book.normalized_reviews_per_year,
                Book.normalized_rating_count,
                Book.normalized_review_count,
                Book.average_review_score,
                Book.title,
                Book.summary
            ).filter(Book.genre != None)

            books = query.all()
            return books

        except Exception as e:
            # If there's an error fetching data, raise it
            raise

    def _organize_books_by_genre(self, books: List[Book]) -> Dict:
        """Group books by their genres."""
        genre_books = defaultdict(list)

        for book in tqdm(books, desc="Processing books"):
            if not self._validate_book_data(book):
                continue

            score = self._calculate_book_score(book)
            genres = parse_genres(book.genre)

            for genre in genres:
                genre_books[genre].append({
                    'isbn13': book.isbn13,
                    'image_location': book.image_location,
                    'score': score,
                    'title': book.title,
                    'summary': book.summary,
                    'is_augmented': False,
                    'genre_tier': {}
                })

        return genre_books

    def _validate_book_data(self, book: Book) -> bool:
        """Ensure that the book has all the necessary data."""
        try:
            if not all([
                book.isbn13,
                book.image_location,
                os.path.exists(book.image_location),
                book.genre,
                book.normalized_ratings_per_year is not None,
                book.normalized_reviews_per_year is not None,
                book.normalized_rating_count is not None,
                book.normalized_review_count is not None,
                book.title,
                book.summary
            ]):
                return False

            # Check if the image format is valid
            if not any(book.image_location.lower().endswith(fmt) for fmt in VALID_IMAGE_FORMATS):
                return False

            return True

        except Exception:
            # If validation fails for any reason, exclude the book
            return False

    def _calculate_book_score(self, book: Book) -> float:
        """Compute a score for the book based on various metrics."""
        try:
            metrics = [
                book.normalized_ratings_per_year,
                book.normalized_reviews_per_year,
                book.normalized_rating_count,
                book.normalized_review_count
            ]

            if book.average_review_score:
                try:
                    normalized_score = float(book.average_review_score)
                    if 0 <= normalized_score <= 5:
                        metrics.append(normalized_score / 5.0)
                except (ValueError, TypeError):
                    pass

            return sum(metrics) / len(metrics)

        except Exception:
            # If scoring fails, assign a default score
            return 0.0

    def _validate_and_filter_genres(self, genre_books: Dict) -> Dict:
        """Filter out genres that don't meet the minimum sample requirement."""
        validated_genres = {}

        for genre, books in genre_books.items():
            original_count = len(books)
            self.genre_stats[genre]['original_count'] = original_count

            if original_count >= self.min_samples / 3:
                validated_genres[genre] = books

        return validated_genres

    def _balance_genres(self, genre_books: Dict) -> Dict:
        """Ensure each genre has a balanced number of samples."""
        balanced_genres = {}

        for genre, books in genre_books.items():
            # Sort books into tiers based on their scores
            tier_books = self._sort_into_tiers(books, genre)

            # Determine how many samples we need
            original_count = len(books)
            target_count = min(self.max_samples, max(self.min_samples, original_count))
            target_per_tier = target_count // 5
            remainder = target_count % 5

            balanced_tier_books = {}
            for tier in range(5):
                tier_target = target_per_tier + (1 if tier < remainder else 0)
                current_tier_books = tier_books.get(str(tier), [])
                current_count = len(current_tier_books)

                if current_count < tier_target:
                    # Need to add more samples through augmentation
                    augmented_books = self._augment_tier_books(
                        current_tier_books,
                        tier_target - current_count,
                        tier,
                        genre
                    )
                    balanced_tier_books[str(tier)] = current_tier_books + augmented_books
                elif current_count > tier_target:
                    # Need to reduce the number of samples
                    balanced_tier_books[str(tier)] = self._reduce_tier_samples(
                        current_tier_books,
                        tier_target
                    )
                else:
                    balanced_tier_books[str(tier)] = current_tier_books

            # Combine all tiers back into the genre
            balanced_genres[genre] = []
            for tier in balanced_tier_books.values():
                balanced_genres[genre].extend(tier)

            # Update statistics
            self.genre_stats[genre]['original_count'] = original_count
            self.genre_stats[genre]['total_count'] = len(balanced_genres[genre])
            for tier, books in balanced_tier_books.items():
                self.genre_stats[genre]['percentile_distribution'][tier] = len(books)

        return balanced_genres

    def _sort_into_tiers(self, books: List[Dict], genre: str) -> Dict[str, List[Dict]]:
        """Divide books into percentile-based tiers."""
        scores = np.array([book['score'] for book in books])
        tier_boundaries = np.percentile(scores, [20, 40, 60, 80])

        tier_books = defaultdict(list)
        for book in books:
            tier = int(np.searchsorted(tier_boundaries, book['score'], side='right'))
            book['genre_tier'][genre] = tier
            tier_books[str(tier)].append(book)

        return tier_books

    def _reduce_tier_samples(self, books: List[Dict], target_count: int) -> List[Dict]:
        """Reduce the number of samples in a tier while maintaining diversity."""
        if target_count >= len(books):
            return books

        books_df = pd.DataFrame(books)
        books_df['score'] = pd.to_numeric(books_df['score'], errors='coerce')

        try:
            books_df['sub_tier'] = pd.qcut(books_df['score'], q=5, labels=False)
        except ValueError:
            try:
                books_df['sub_tier'] = pd.qcut(books_df['score'], q=5, labels=False, duplicates='drop')
            except ValueError:
                books_df['rank'] = books_df['score'].rank(method='first')
                books_df['sub_tier'] = pd.qcut(books_df['rank'], q=5, labels=False)

        samples_per_subtier = target_count // 5
        remainder = target_count % 5

        selected_books = []
        unique_subtiers = books_df['sub_tier'].dropna().unique()
        num_subtiers = len(unique_subtiers)

        if num_subtiers < 5:
            samples_per_subtier = target_count // num_subtiers
            remainder = target_count % num_subtiers

        for i, subtier in enumerate(sorted(unique_subtiers)):
            subtier_books = books_df[books_df['sub_tier'] == subtier]
            subtier_target = samples_per_subtier + (1 if i < remainder else 0)

            if len(subtier_books) <= subtier_target:
                selected_books.extend(subtier_books.to_dict('records'))
            else:
                sampled = subtier_books.sample(n=subtier_target, random_state=42)
                selected_books.extend(sampled.to_dict('records'))

        # If not enough samples, add some randomly
        if len(selected_books) < target_count:
            remaining_needed = target_count - len(selected_books)
            available_books = [b for b in books if b not in selected_books]
            if available_books:
                additional_books = random.sample(available_books, min(remaining_needed, len(available_books)))
                selected_books.extend(additional_books)

        return selected_books

    def _augment_tier_books(self, source_books: List[Dict], num_needed: int, tier: int, genre: str) -> List[Dict]:
        """Generate additional samples through augmentation to meet target count."""
        augmented_books = []
        books_cycle = cycle(source_books)

        while len(augmented_books) < num_needed:
            book = next(books_cycle)
            try:
                with Image.open(book['image_location']) as img:
                    if not self.augmenter.is_valid_image(img):
                        continue

                    geometric_augmented = self.augmenter.apply_geometric_augmentation(img)
                    color_augmented = self.augmenter.apply_color_augmentation(img)

                    for aug_type, aug_img in geometric_augmented + color_augmented:
                        if len(augmented_books) >= num_needed:
                            break

                        aug_path = self._save_augmented_image(aug_img, book['isbn13'], aug_type)

                        augmented_book = {
                            'isbn13': f"{book['isbn13']}_aug_{aug_type}",
                            'image_location': aug_path,
                            'score': book['score'],
                            'genre_tier': {genre: tier},
                            'title': book['title'],
                            'summary': book['summary'],
                            'is_augmented': True
                        }
                        augmented_books.append(augmented_book)

            except Exception:
                # Skip any books that cause errors during augmentation
                continue

        return augmented_books

    def _save_augmented_image(self, image: Image.Image, isbn13: str, aug_type: str) -> str:
        """Save the augmented image to disk."""
        try:
            aug_dir = os.path.join('augmented_images', isbn13)
            os.makedirs(aug_dir, exist_ok=True)

            aug_path = os.path.join(aug_dir, f"{aug_type}.jpg")
            image.save(aug_path, 'JPEG', quality=95)

            return aug_path

        except Exception:
            # If saving fails, skip this augmentation
            raise

    def _create_genre_map(self, balanced_data: Dict) -> Dict[str, int]:
        """Map each genre to a unique index."""
        return {genre: idx for idx, genre in enumerate(balanced_data.keys())}

    def _calculate_percentile_tiers(self, balanced_data: Dict[str, List[Dict]],
                                    genre_map: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Assign each book to its percentile tier within its genre."""
        genre_percentiles = {str(genre): {} for genre in genre_map.keys()}

        for genre, books in balanced_data.items():
            for book in books:
                genre_percentiles[str(genre)][str(book['isbn13'])] = book['genre_tier'].get(genre)

        return genre_percentiles

    def _prepare_final_dataset(self, balanced_data: Dict, genre_map: Dict,
                               percentile_tiers: Dict) -> List[Tuple]:
        """Compile the final dataset ready for training."""
        final_data = []

        for genre, books in balanced_data.items():
            genre_idx = genre_map[genre]
            for book in books:
                final_data.append((
                    book['isbn13'],
                    book['image_location'],
                    genre_idx,
                    book['genre_tier'][genre],
                    book['title'],
                    book['summary']
                ))

        random.shuffle(final_data)
        return final_data


class AugmentedBookCoverDataset(Dataset):
    """Dataset class that handles image loading, augmentation, and text tokenization."""

    def __init__(self, data: List[Tuple], tokenizer: DebertaTokenizer, max_length: int = 256,
                 augment: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmenter = ImageAugmenter() if augment else None

        self.base_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224)
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=models.EfficientNet_B7_Weights.IMAGENET1K_V1.transforms().mean,
                std=models.EfficientNet_B7_Weights.IMAGENET1K_V1.transforms().std
            )
        ])

        # Keep track of loading stats
        self.successful_loads = 0
        self.failed_loads = 0
        self.error_types = defaultdict(int)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Tuple]:
        isbn13, image_path, genre_index, percentile_tier_index, title, summary = self.data[idx]

        try:
            # Open and process the image
            with Image.open(image_path) as img:
                image = img.convert('RGB')
                image = self.base_transform(image)

                # Apply augmentation if enabled
                if self.augment and random.random() < 0.5:
                    if random.random() < 0.5:
                        augmented = self.augmenter.apply_geometric_augmentation(image)
                    else:
                        augmented = self.augmenter.apply_color_augmentation(image)

                    if augmented:
                        image = augmented[0][1]  # Use the first valid augmentation

                image = self.normalize(image)

            # Combine title and summary for text processing
            text = (title or '') + ' ' + (summary or '')
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.successful_loads += 1
            return isbn13, image, genre_index, percentile_tier_index, encoding['input_ids'].squeeze(0), encoding[
                'attention_mask'].squeeze(0)

        except (IOError, UnidentifiedImageError) as e:
            # If there's an error loading the image, skip this sample
            error_type = type(e).__name__
            self.error_types[error_type] += 1
            self.failed_loads += 1
            return None

    def get_loading_stats(self) -> Dict[str, Any]:
        """Provide statistics on dataset loading."""
        return {
            'successful_loads': self.successful_loads,
            'failed_loads': self.failed_loads,
            'total_attempts': self.successful_loads + self.failed_loads,
            'success_rate': self.successful_loads / (self.successful_loads + self.failed_loads) * 100,
            'error_types': dict(self.error_types)
        }


class GenreMetrics:
    """Class to compute and store various metrics for each genre."""

    def __init__(self, num_genres: int, num_percentiles: int):
        self.num_genres = num_genres
        self.num_percentiles = num_percentiles
        self.reset()

    def reset(self):
        """Reset all stored metrics."""
        self.genre_confusion_matrices = {str(i): np.zeros((self.num_percentiles, self.num_percentiles))
                                         for i in range(self.num_genres)}
        self.genre_correct = np.zeros(self.num_genres)
        self.genre_total = np.zeros(self.num_genres)
        self.genre_predictions = defaultdict(list)
        self.genre_targets = defaultdict(list)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               genre_indices: torch.Tensor):
        """Update metrics with the latest batch of predictions."""
        for pred, target, genre_idx in zip(predictions, targets, genre_indices):
            genre_key = str(genre_idx.item())
            target_idx = int(target.item())
            pred_idx = int(pred.item())

            self.genre_confusion_matrices[genre_key][target_idx, pred_idx] += 1
            self.genre_total[int(genre_key)] += 1

            if pred_idx == target_idx:
                self.genre_correct[int(genre_key)] += 1

            self.genre_predictions[genre_key].append(pred_idx)
            self.genre_targets[genre_key].append(target_idx)

    def _calculate_macro_precision(self, confusion_matrix: np.ndarray) -> float:
        """Compute macro-averaged precision."""
        precisions = []
        for i in range(self.num_percentiles):
            col_sum = confusion_matrix[:, i].sum()
            if col_sum > 0:
                precision = confusion_matrix[i, i] / col_sum
                precisions.append(precision)
        return np.mean(precisions) if precisions else 0.0

    def _calculate_macro_recall(self, confusion_matrix: np.ndarray) -> float:
        """Compute macro-averaged recall."""
        recalls = []
        for i in range(self.num_percentiles):
            row_sum = confusion_matrix[i, :].sum()
            if row_sum > 0:
                recall = confusion_matrix[i, i] / row_sum
                recalls.append(recall)
        return np.mean(recalls) if recalls else 0.0

    def _calculate_weighted_f1(self, confusion_matrix: np.ndarray) -> float:
        """Compute weighted F1 score."""
        f1_scores = []
        weights = []

        for i in range(self.num_percentiles):
            row_sum = confusion_matrix[i, :].sum()
            if row_sum > 0:
                col_sum = confusion_matrix[:, i].sum()
                if col_sum > 0:
                    precision = confusion_matrix[i, i] / col_sum
                    recall = confusion_matrix[i, i] / row_sum

                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        f1_scores.append(f1)
                        weights.append(row_sum)

        return np.average(f1_scores, weights=weights) if f1_scores else 0.0

    def _calculate_adjacent_accuracy(self, predictions: List[int], targets: List[int]) -> float:
        """Compute accuracy allowing for off-by-one predictions."""
        if not predictions or not targets:
            return 0.0
        adjacent_correct = sum(1 for pred, target in zip(predictions, targets)
                               if abs(pred - target) <= 1)
        return adjacent_correct / len(predictions)

    def compute(self) -> Tuple[Dict[int, float], Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
        """Calculate all metrics for each genre."""
        genre_accuracies = {}
        genre_per_class_accuracies = {}
        genre_detailed_metrics = {}

        for genre_idx in range(self.num_genres):
            genre_key = str(genre_idx)
            if genre_key in self.genre_confusion_matrices and self.genre_total[genre_idx] > 0:
                confusion_matrix = self.genre_confusion_matrices[genre_key]

                # Per-class accuracy
                row_sums = np.sum(confusion_matrix, axis=1)
                per_class_acc = np.zeros(self.num_percentiles)
                for i in range(self.num_percentiles):
                    if row_sums[i] > 0:
                        per_class_acc[i] = confusion_matrix[i, i] / row_sums[i]

                accuracy = self.genre_correct[genre_idx] / self.genre_total[genre_idx]

                genre_accuracies[genre_idx] = accuracy
                genre_per_class_accuracies[genre_idx] = per_class_acc

                # Detailed metrics
                if genre_key in self.genre_predictions:
                    genre_detailed_metrics[genre_idx] = {
                        'accuracy': accuracy,
                        'macro_precision': self._calculate_macro_precision(confusion_matrix),
                        'macro_recall': self._calculate_macro_recall(confusion_matrix),
                        'weighted_f1': self._calculate_weighted_f1(confusion_matrix),
                        'adjacent_accuracy': self._calculate_adjacent_accuracy(
                            self.genre_predictions[genre_key],
                            self.genre_targets[genre_key]
                        )
                    }

        return genre_accuracies, genre_per_class_accuracies, genre_detailed_metrics


def train_with_metrics(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                       num_epochs: int, learning_rate: float, device: torch.device,
                       num_genres: int) -> Dict:
    """Train the model while keeping track of various performance metrics."""

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    genre_metrics = GenreMetrics(num_genres, num_percentiles)
    training_history = defaultdict(list)
    best_val_accuracy = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        genre_metrics.reset()

        for batch in tqdm(train_loader, desc="Training"):
            if not batch:
                continue

            train_steps += 1
            _, images, genre_indices, percentile_tier_indices, input_ids, attention_masks = batch

            # Move tensors to the appropriate device
            images = images.to(device)
            genre_indices = genre_indices.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            percentile_tier_indices = percentile_tier_indices.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, genre_indices, input_ids, attention_masks)
            loss = criterion(outputs, percentile_tier_indices)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            # Update metrics
            _, predicted = torch.max(outputs.data, 1)
            genre_metrics.update(predicted.cpu(), percentile_tier_indices.cpu(), genre_indices.cpu())

        # Calculate average training loss
        avg_train_loss = total_train_loss / train_steps
        train_genre_accuracies, train_per_class_accuracies, train_detailed_metrics = genre_metrics.compute()

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_steps = 0
        genre_metrics.reset()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if not batch:
                    continue

                val_steps += 1
                _, images, genre_indices, percentile_tier_indices, input_ids, attention_masks = batch

                # Move tensors to the appropriate device
                images = images.to(device)
                genre_indices = genre_indices.to(device)
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                percentile_tier_indices = percentile_tier_indices.to(device)

                # Forward pass
                outputs = model(images, genre_indices, input_ids, attention_masks)
                loss = criterion(outputs, percentile_tier_indices)

                total_val_loss += loss.item()

                # Update metrics
                _, predicted = torch.max(outputs.data, 1)
                genre_metrics.update(predicted.cpu(), percentile_tier_indices.cpu(), genre_indices.cpu())

        # Calculate average validation loss
        avg_val_loss = total_val_loss / val_steps
        val_genre_accuracies, val_per_class_accuracies, val_detailed_metrics = genre_metrics.compute()

        # Compute overall validation accuracy
        overall_val_accuracy = sum(val_genre_accuracies.values()) / len(val_genre_accuracies)

        # Check if this is the best model so far
        if overall_val_accuracy > best_val_accuracy:
            best_val_accuracy = overall_val_accuracy
            patience_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'genre_metrics': val_detailed_metrics
            }, 'BestModel.pth')
        else:
            patience_counter += 1

        # Print epoch results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        print("\nTraining Metrics by Genre:")
        for genre_idx, metrics in train_detailed_metrics.items():
            print(f"\nGenre {genre_idx}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        print("\nValidation Metrics by Genre:")
        for genre_idx, metrics in val_detailed_metrics.items():
            print(f"\nGenre {genre_idx}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        # Record metrics
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(overall_val_accuracy)
        training_history['train_metrics'].append(train_detailed_metrics)
        training_history['val_metrics'].append(val_detailed_metrics)

        # Update the learning rate
        scheduler.step()

        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    return training_history


def evaluate_model(model, test_loader, criterion, device, genre_metrics=None):
    """Assess the model's performance on the test set."""

    model.eval()
    total_test_loss = 0
    total_correct = 0
    total_samples = 0
    confusion_matrix = torch.zeros(num_percentiles, num_percentiles)

    if genre_metrics:
        genre_metrics.reset()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if not batch:
                continue

            _, images, genre_indices, percentile_tier_indices, input_ids, attention_masks = batch

            # Move tensors to the appropriate device
            images = images.to(device)
            genre_indices = genre_indices.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            percentile_tier_indices = percentile_tier_indices.to(device)

            # Forward pass
            outputs = model(images, genre_indices, input_ids, attention_masks)
            loss = criterion(outputs, percentile_tier_indices)
            total_test_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Count correct predictions
            correct = torch.eq(predicted, percentile_tier_indices).sum().item()
            total_correct += correct
            total_samples += percentile_tier_indices.size(0)

            # Update confusion matrix
            for t, p in zip(percentile_tier_indices.cpu(), predicted.cpu()):
                confusion_matrix[t.long(), p.long()] += 1

            if genre_metrics:
                genre_metrics.update(predicted.cpu(), percentile_tier_indices.cpu(), genre_indices.cpu())

    # Calculate average test loss and accuracy
    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / total_samples

    print("\nOverall Confusion Matrix:")
    print(confusion_matrix)
    per_class_accuracies = confusion_matrix.diag() / confusion_matrix.sum(1)
    print("\nOverall Per-class Accuracies:")
    for i, acc in enumerate(per_class_accuracies):
        percentile_range = f"{i * 20}-{(i + 1) * 20}" if i < 4 else "80-100"
        print(f"Class {i} ({percentile_range}%): {acc:.4f}")

    if genre_metrics:
        genre_accuracies, genre_per_class_accuracies, genre_detailed_metrics = genre_metrics.compute()
        print("\nPer-Genre Metrics:")
        for genre_idx, metrics in genre_detailed_metrics.items():
            print(f"\nGenre {genre_idx}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

    return avg_test_loss, test_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train a balanced book percentile predictor")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=12, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="model_outputs", help="Directory for saving outputs")
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and balance the dataset
    data_manager = GenreBalancedDataManager()
    data, genre_map, percentile_tiers = data_manager.load_and_balance_data()

    # Save the genre mappings for future reference
    with open(os.path.join(args.output_dir, 'genre_map.json'), 'w') as f:
        json.dump(genre_map, f, indent=4)

    # Split the data into training, validation, and test sets
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    # Initialize the tokenizer
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

    # Create dataset instances
    train_dataset = AugmentedBookCoverDataset(train_data, tokenizer=tokenizer, augment=True)
    val_dataset = AugmentedBookCoverDataset(val_data, tokenizer=tokenizer, augment=False)
    test_dataset = AugmentedBookCoverDataset(test_data, tokenizer=tokenizer, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=collate_fn, num_workers=4)

    # Initialize the model
    model = BookPercentilePredictor(num_genres=len(genre_map), num_percentiles=num_percentiles)

    # Start training
    training_history = train_with_metrics(
        model, train_loader, val_loader, args.num_epochs,
        args.learning_rate, device, len(genre_map)
    )

    # Save the training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=4)

    # Load the best model saved during training
    best_model_state = torch.load('BestModel.pth')
    model.load_state_dict(best_model_state['model_state_dict'])

    # Evaluate the model on the test set
    print("\nPerforming final evaluation...")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    genre_metrics = GenreMetrics(len(genre_map), num_percentiles)
    avg_test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device, genre_metrics)

    print(f"\nFinal Test Results:")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Overall Test Accuracy: {test_accuracy:.4f}")

    # Save the final model and related information
    final_save_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'genre_map': genre_map,
        'num_genres': len(genre_map),
        'num_percentiles': num_percentiles,
        'training_history': training_history,
        'test_metrics': {
            'loss': avg_test_loss,
            'accuracy': test_accuracy
        }
    }, final_save_path)

    print(f"\nTraining complete. Model and metadata saved to {args.output_dir}")


if __name__ == "__main__":
    main()
