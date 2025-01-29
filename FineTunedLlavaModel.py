from typing import Optional, Dict, List, Tuple
from transformers import AutoProcessor, AutoModelForCausalLM
import json
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from common import get_session, Book
from collections import defaultdict
from itertools import cycle
import random
import os
import numpy as np
from tqdm import tqdm
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    def __init__(self):
        self.augmentation_stats = defaultdict(int)
        self.failed_augmentations = defaultdict(int)

    @staticmethod
    def is_valid_image(image: Image.Image) -> bool:
        try:
            if image.size[0] < 64 or image.size[1] < 64:
                return False
            tensor_image = TF.to_tensor(image)
            if torch.allclose(tensor_image, tensor_image[0, 0, 0]):
                return False
            return True
        except Exception as e:
            logger.error(f"Image validation error: {str(e)}")
            return False

    def apply_geometric_augmentation(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        augmented_images = []
        try:
            tensor_image = TF.to_tensor(image)
            # Horizontal flip
            flipped = TF.hflip(tensor_image)
            pil_flipped = TF.to_pil_image(flipped)
            if self.is_valid_image(pil_flipped):
                augmented_images.append(('flip', pil_flipped))
                self.augmentation_stats['flip'] += 1

            # Rotations
            for angle in [-15, -10, -5, 5, 10, 15]:
                rotated = TF.rotate(tensor_image, angle)
                pil_rotated = TF.to_pil_image(rotated)
                if self.is_valid_image(pil_rotated):
                    augmented_images.append((f'rotate_{angle}', pil_rotated))
                    self.augmentation_stats[f'rotate_{angle}'] += 1

            # Scale variations
            for scale in [0.9, 1.1]:
                h, w = tensor_image.shape[-2:]
                new_h = int(h * scale)
                new_w = int(w * scale)
                resized = TF.resize(tensor_image, size=[new_h, new_w])
                if scale < 1:  # Zoom out - pad
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    padded = TF.pad(resized, [pad_h, pad_w], fill=1)
                    if padded.shape[-2:] != (h, w):
                        padded = TF.center_crop(padded, [h, w])
                    zoomed = padded
                else:  # Zoom in - crop
                    zoomed = TF.center_crop(resized, [h, w])

                pil_zoomed = TF.to_pil_image(zoomed)
                if self.is_valid_image(pil_zoomed):
                    augmented_images.append((f'zoom_{scale}', pil_zoomed))
                    self.augmentation_stats[f'zoom_{scale}'] += 1

        except Exception as e:
            logger.error(f"Geometric augmentation error: {str(e)}")
            self.failed_augmentations['geometric'] += 1
            return []
        return augmented_images

    def apply_color_augmentation(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        augmented_images = []
        try:
            tensor_image = TF.to_tensor(image)
            for factor in [0.8, 1.2]:
                # Brightness
                brightened = TF.adjust_brightness(tensor_image, factor)
                pil_brightened = TF.to_pil_image(brightened)
                if self.is_valid_image(pil_brightened):
                    augmented_images.append((f'brightness_{factor}', pil_brightened))
                    self.augmentation_stats[f'brightness_{factor}'] += 1

                # Contrast
                contrasted = TF.adjust_contrast(tensor_image, factor)
                pil_contrasted = TF.to_pil_image(contrasted)
                if self.is_valid_image(pil_contrasted):
                    augmented_images.append((f'contrast_{factor}', pil_contrasted))
                    self.augmentation_stats[f'contrast_{factor}'] += 1

                # Saturation
                saturated = TF.adjust_saturation(tensor_image, factor)
                pil_saturated = TF.to_pil_image(saturated)
                if self.is_valid_image(pil_saturated):
                    augmented_images.append((f'saturation_{factor}', pil_saturated))
                    self.augmentation_stats[f'saturation_{factor}'] += 1

        except Exception as e:
            logger.error(f"Color augmentation error: {str(e)}")
            self.failed_augmentations['color'] += 1
            return []
        return augmented_images


class GenreBalancedDataManager:
    def __init__(self, sample_number: int):
        self.sample_number = sample_number
        self.min_samples = sample_number // 3
        self.augmenter = ImageAugmenter()
        self.genre_stats = defaultdict(lambda: {
            'original_count': 0,
            'augmented_count': 0,
            'total_count': 0
        })
        self.all_genres = set()

    def load_and_balance_data(self) -> List[Dict]:
        logger.info("Starting data loading and balancing process...")
        session = get_session()
        books = self._load_books_from_db(session)
        session.close()

        genre_books = self._organize_books_by_genre(books)
        validated_genre_books = self._validate_and_filter_genres(genre_books)
        balanced_data = self._balance_genres(validated_genre_books)
        self._log_balancing_stats(balanced_data)
        return self._prepare_final_dataset(balanced_data)

    def _load_books_from_db(self, session) -> List[Book]:
        logger.info("Loading books from database...")
        try:
            query = session.query(
                Book.isbn13,
                Book.image_location,
                Book.genre
            ).filter(Book.genre != None)
            books = query.all()
            logger.info(f"Successfully loaded {len(books)} books from database")
            return books
        except Exception as e:
            logger.error(f"Database loading error: {str(e)}")
            raise

    def _organize_books_by_genre(self, books: List[Book]) -> Dict:
        logger.info("Organizing books by genre...")
        genre_books = defaultdict(list)
        for book in tqdm(books, desc="Processing books"):
            if not self._validate_book_data(book):
                continue
            genres = self._parse_genres(book.genre)
            for genre in genres:
                genre_books[genre].append({
                    'isbn13': book.isbn13,
                    'image_location': book.image_location,
                    'genres': genres,
                    'is_augmented': False
                })
        return genre_books

    def _validate_book_data(self, book: Book) -> bool:
        try:
            if not all([
                book.isbn13,
                book.image_location,
                os.path.exists(book.image_location),
                book.genre
            ]):
                return False
            return True
        except Exception as e:
            logger.error(f"Book validation error for {book.isbn13}: {str(e)}")
            return False

    def _parse_genres(self, genre_str: str) -> List[str]:
        try:
            if isinstance(genre_str, str):
                try:
                    genres = json.loads(genre_str)
                except json.JSONDecodeError:
                    genres = [g.strip().strip('"').strip("'") for g in genre_str.split(',')]
                genres = [g.lower() for g in genres if g]
                self.all_genres.update(genres)
                return genres
            return []
        except Exception as e:
            logger.error(f"Genre parsing error: {str(e)}")
            return []

    def _validate_and_filter_genres(self, genre_books: Dict) -> Dict:
        logger.info("Validating and filtering genres...")
        validated_genres = {}
        for genre, books in genre_books.items():
            original_count = len(books)
            self.genre_stats[genre]['original_count'] = original_count
            if original_count >= self.min_samples:
                validated_genres[genre] = books
                logger.info(f"Genre {genre}: {original_count} original samples")
        return validated_genres

    def _balance_genres(self, genre_books: Dict) -> Dict:
        logger.info("Balancing genres...")
        balanced_genres = {}
        for genre, books in genre_books.items():
            original_count = len(books)
            if original_count > self.sample_number:
                balanced_books = self._reduce_samples(books, self.sample_number)
                logger.info(f"Reduced {genre} from {original_count} to {len(balanced_books)} samples")
            else:
                balanced_books = books + self._augment_samples(
                    books,
                    self.sample_number - original_count
                )
                logger.info(f"Augmented {genre} from {original_count} to {len(balanced_books)} samples")
            balanced_genres[genre] = balanced_books
        return balanced_genres

    def _reduce_samples(self, books: List[Dict], target_count: int) -> List[Dict]:
        if target_count >= len(books):
            return books
        return random.sample(books, target_count)

    def _augment_samples(self, source_books: List[Dict], num_needed: int) -> List[Dict]:
        augmented_books = []
        books_cycle = cycle(source_books)
        while len(augmented_books) < num_needed:
            book = next(books_cycle)
            try:
                with Image.open(book['image_location']) as img:
                    if not self.augmenter.is_valid_image(img):
                        continue
                    augmentations = (
                        self.augmenter.apply_geometric_augmentation(img) +
                        self.augmenter.apply_color_augmentation(img)
                    )
                    for aug_type, aug_img in augmentations:
                        if len(augmented_books) >= num_needed:
                            break
                        aug_path = self._save_augmented_image(aug_img, book['isbn13'], aug_type)
                        augmented_book = {
                            'isbn13': f"{book['isbn13']}_aug_{aug_type}",
                            'image_location': aug_path,
                            'genres': book['genres'],
                            'is_augmented': True
                        }
                        augmented_books.append(augmented_book)
            except Exception as e:
                logger.error(f"Augmentation error for {book['isbn13']}: {str(e)}")
                continue
        return augmented_books

    def _save_augmented_image(self, image: Image.Image, isbn13: str, aug_type: str) -> str:
        try:
            aug_dir = os.path.join('augmented_images', isbn13)
            os.makedirs(aug_dir, exist_ok=True)
            aug_path = os.path.join(aug_dir, f"{aug_type}.jpg")
            image.save(aug_path, 'JPEG', quality=95)
            return aug_path
        except Exception as e:
            logger.error(f"Error saving augmented image: {str(e)}")
            raise

    def _log_balancing_stats(self, balanced_data: Dict):
        logger.info("\nFinal Data Distribution Statistics:")
        for genre, books in balanced_data.items():
            logger.info(f"\nGenre: {genre}")
            total_books = len(books)
            augmented_books = sum(1 for book in books if book['is_augmented'])
            original_books = total_books - augmented_books
            logger.info(f"Total books: {total_books}")
            logger.info(f"Original books: {original_books}")
            logger.info(f"Augmented books: {augmented_books}")

    def _prepare_final_dataset(self, balanced_data: Dict) -> List[Dict]:
        final_data = []
        for genre, books in balanced_data.items():
            for book in books:
                final_data.append({
                    'isbn13': book['isbn13'],
                    'image_location': book['image_location'],
                    'genres': book['genres']
                })
        random.shuffle(final_data)
        return final_data


class LLaVADataset(Dataset):
    def __init__(self, data: List[Dict], processor: AutoProcessor):
        self.data = data
        self.processor = processor
        self.examples = []
        for item in data:
            self.examples.append(self._prepare_example(item))

    def _prepare_example(self, book: Dict) -> Dict:
        prompt = (
            "USER: This is a book cover image. What genres does this book belong to? List only the genres, separated by commas.\n\n"
            "ASSISTANT: "
        )
        target_text = ", ".join(book['genres'])
        full_text = prompt + target_text

        # Use processor.tokenizer for text
        text_inputs = self.processor.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True
        )

        # Prompt length
        prompt_inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True
        )
        prompt_length = prompt_inputs.input_ids.shape[1]

        return {
            'isbn13': book['isbn13'],
            'input_ids': text_inputs.input_ids[0],
            'attention_mask': text_inputs.attention_mask[0],
            'target_text': target_text,
            'prompt_length': prompt_length,
            'image_location': book['image_location'],
            'true_genres': book['genres']
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        example = self.examples[idx]
        try:
            with Image.open(example['image_location']) as img:
                image = img.convert('RGB')
                # Use processor.vision_processor for images
                image_inputs = self.processor.image_processor(
                    image,
                    return_tensors="pt"
                )

                return {
                    'isbn13': example['isbn13'],
                    'input_ids': example['input_ids'],
                    'attention_mask': example['attention_mask'],
                    'images': image_inputs.pixel_values[0],  # rename to 'images'
                    'true_genres': example['true_genres'],
                    'prompt_length': example['prompt_length']
                }
        except (IOError, OSError, UnidentifiedImageError) as e:
            logger.error(f"Error loading image for {example['isbn13']}: {str(e)}")
            return None


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_length = max(item['input_ids'].shape[0] for item in batch)

    input_ids_padded = []
    attention_mask_padded = []
    images = []
    prompt_lengths = []
    isbn13_list = []
    true_genres_list = []

    for item in batch:
        pad_len = max_length - item['input_ids'].shape[0]
        padded_input_ids = torch.cat([item['input_ids'], torch.full((pad_len,), fill_value=0, dtype=torch.long)])
        padded_attention_mask = torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])

        input_ids_padded.append(padded_input_ids)
        attention_mask_padded.append(padded_attention_mask)
        images.append(item['images'])
        prompt_lengths.append(item['prompt_length'])
        isbn13_list.append(item['isbn13'])
        true_genres_list.append(item['true_genres'])

    return {
        'isbn13': isbn13_list,
        'input_ids': torch.stack(input_ids_padded),
        'attention_mask': torch.stack(attention_mask_padded),
        'images': torch.stack(images),  # renamed key to 'images'
        'prompt_length': torch.tensor(prompt_lengths),
        'true_genres': true_genres_list
    }


def evaluate_predictions(predictions: List[str], true_genres: List[List[str]]) -> Dict[str, float]:
    metrics = {
        'exact_match': 0,
        'partial_match': 0,
        'total_predictions': len(predictions)
    }

    for pred, true in zip(predictions, true_genres):
        pred_set = set(g.strip().lower() for g in pred.split(',') if g.strip())
        true_set = set(g.strip().lower() for g in true if g.strip())

        if pred_set == true_set:
            metrics['exact_match'] += 1
        elif len(pred_set.intersection(true_set)) > 0:
            metrics['partial_match'] += 1

    total = metrics['total_predictions']
    if total > 0:
        metrics['exact_match_rate'] = metrics['exact_match'] / total * 100
        metrics['partial_match_rate'] = metrics['partial_match'] / total * 100

    return metrics


def main(sample_number: int = 35000, num_train_epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_id = "liuhaotian/llava-v1.5-7b"
    logger.info("Loading LLaVA-v1.5-7b model and processor...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    max_length = 512
    processor.tokenizer.model_max_length = max_length

    logger.info(f"Initializing data manager with {sample_number} samples per genre...")
    data_manager = GenreBalancedDataManager(sample_number)
    balanced_data = data_manager.load_and_balance_data()

    genre_counts = defaultdict(int)
    for item in balanced_data:
        for g in item['genres']:
            genre_counts[g] += 1
    min_genre_count = min(genre_counts.values())
    if min_genre_count < 1000:
        logger.warning(f"Warning: Some genres have very few samples ({min_genre_count})")

    n_samples = len(balanced_data)
    train_idx = int(0.7 * n_samples)
    val_idx = int(0.8 * n_samples)

    train_data = balanced_data[:train_idx]
    val_data = balanced_data[train_idx:val_idx]
    test_data = balanced_data[val_idx:]

    train_dataset = LLaVADataset(train_data, processor)
    val_dataset = LLaVADataset(val_data, processor)
    test_dataset = LLaVADataset(test_data, processor)

    # Try initial batch size and reduce if OOM
    batch_size = 4
    try:
        _ = next(iter(DataLoader(train_dataset, batch_size=batch_size)))
    except RuntimeError:
        batch_size = 2
        logger.warning("Reducing batch size to 2 due to GPU memory constraints")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_train_epochs * len(train_loader)
    )
    grad_accum_steps = 2
    scaler = torch.cuda.amp.GradScaler()

    best_val_exact = 0.0
    best_epoch = 0

    model.train()
    logger.info("Starting fine-tuning...")
    for epoch in range(num_train_epochs):
        epoch_loss = 0.0
        model.train()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
            if batch is None:
                continue
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['images'].to(device)
            prompt_lengths = batch['prompt_length'].to(device)

            labels = input_ids.clone()
            for idx, p_len in enumerate(prompt_lengths):
                labels[idx, :p_len] = -100

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,  # Use images instead of pixel_values
                    labels=labels
                )
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps

            if (i + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")

        # Validation
        model.eval()
        val_metrics = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                if batch is None:
                    continue

                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'images': batch['images'].to(device)
                }

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )

                predictions = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_metrics = evaluate_predictions(predictions, batch['true_genres'])
                val_metrics.append(batch_metrics)

        val_exact = np.mean([m['exact_match_rate'] for m in val_metrics if 'exact_match_rate' in m])
        logger.info(f"Validation Exact Match Rate: {val_exact:.2f}%")

        if val_exact > best_val_exact:
            best_val_exact = val_exact
            best_epoch = epoch
            logger.info(f"New best validation exact match rate: {val_exact:.2f}%")

            checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_exact': val_exact,
            }, checkpoint_path)

        # Early stopping if no improvement in 3 epochs
        if epoch - best_epoch >= 3:
            logger.info("Early stopping triggered")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load best checkpoint for final evaluation
    best_checkpoint = torch.load(f'checkpoint_epoch_{best_epoch + 1}.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Test evaluation
    logger.info("Performing final evaluation...")
    model.eval()
    test_metrics = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None:
                continue

            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'images': batch['images'].to(device)
            }

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

            predictions = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_metrics = evaluate_predictions(predictions, batch['true_genres'])
            test_metrics.append(batch_metrics)

    final_metrics = {
        'exact_match_rate': np.mean([m['exact_match_rate'] for m in test_metrics if 'exact_match_rate' in m]),
        'partial_match_rate': np.mean([m['partial_match_rate'] for m in test_metrics if 'partial_match_rate' in m])
    }

    logger.info("\nFinal Test Metrics:")
    logger.info(f"Exact Match Rate: {final_metrics['exact_match_rate']:.2f}%")
    logger.info(f"Partial Match Rate: {final_metrics['partial_match_rate']:.2f}%")

    with open('final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune book genre classifier with balanced genres")
    parser.add_argument("--sample_number", type=int, default=35000,
                        help="Target number of samples per genre")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    args = parser.parse_args()

    main(args.sample_number, args.num_train_epochs)
