import gzip
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests
from transformers import DistilBertTokenizerFast

GENRE_URL_DICT: Dict[str, str] = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


def load_reviews(url: str, head: Optional[int] = 10000, sample_size: int = 2000) -> List[str]:
    """Load reviews from a gzipped JSON stream and return a random sample."""
    reviews: List[str] = []
    count = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            row = json.loads(line)
            review_text = row.get('review_text')
            if review_text:
                reviews.append(review_text)
            count += 1
            if head is not None and count >= head:
                break

    if not reviews:
        raise ValueError(f'No reviews were loaded from {url}')

    return random.sample(reviews, min(sample_size, len(reviews)))


def build_genre_reviews(
    genre_url_dict: Dict[str, str] = GENRE_URL_DICT,
    head: Optional[int] = 10000,
    sample_size: int = 2000,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Download or sample reviews for each genre."""
    random.seed(seed)
    genre_reviews: Dict[str, List[str]] = {}
    for genre, url in genre_url_dict.items():
        genre_reviews[genre] = load_reviews(url, head=head, sample_size=sample_size)
    return genre_reviews


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open('rb') as handle:
        return pickle.load(handle)


def split_train_test(
    genre_reviews_dict: Dict[str, List[str]],
    samples_per_genre: int = 1000,
    train_count: int = 800,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Split reviews into train/test texts and labels."""
    random.seed(seed)
    train_texts: List[str] = []
    train_labels: List[str] = []
    test_texts: List[str] = []
    test_labels: List[str] = []

    for genre, reviews in genre_reviews_dict.items():
        sample_size = min(samples_per_genre, len(reviews))
        sampled_reviews = random.sample(reviews, sample_size)
        train_split = min(train_count, len(sampled_reviews))
        for review in sampled_reviews[:train_split]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled_reviews[train_split:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def encode_texts(
    tokenizer: Union[str, DistilBertTokenizerFast],
    texts: Sequence[str],
    max_length: int = 512,
) -> Dict[str, Any]:
    """Tokenize a list of texts for BERT input."""
    if isinstance(tokenizer, str):
        tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer)
    return tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download and prepare Goodreads genre review data.')
    parser.add_argument('--output_pickle', default='genre_reviews_dict.pickle')
    parser.add_argument('--head', type=int, default=10000)
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    genre_reviews = build_genre_reviews(head=args.head, sample_size=args.sample_size, seed=args.seed)
    save_pickle(genre_reviews, args.output_pickle)
    print(f'Saved genre review dictionary to {args.output_pickle}')
