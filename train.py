import argparse
import os
from pathlib import Path

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments

from data import GENRE_URL_DICT, build_genre_reviews, encode_texts, load_pickle, save_pickle, split_train_test
from utils import MyDataset, compute_metrics, get_label_maps, save_json

os.environ['WANDB_DISABLED'] = 'true'


def load_or_build_genre_reviews(pickle_path: Path, force_download: bool = False) -> dict:
    if pickle_path.exists() and not force_download:
        return load_pickle(pickle_path)
    genre_reviews = build_genre_reviews(genre_url_dict=GENRE_URL_DICT)
    save_pickle(genre_reviews, pickle_path)
    return genre_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a BERT classifier for Goodreads genre review classification.')
    # Keeping params to default  
    parser.add_argument('--dataset_pickle', default='genre_reviews_dict.pickle')
    parser.add_argument('--model_name', default='distilbert-base-cased')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=10)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--head', type=int, default=10000)
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--samples_per_genre', type=int, default=1000)
    parser.add_argument('--train_count', type=int, default=800)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--force_download', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = Path(args.dataset_pickle)
    genre_reviews = load_or_build_genre_reviews(pickle_path, force_download=args.force_download)

    train_texts, train_labels, test_texts, test_labels = split_train_test(
        genre_reviews_dict=genre_reviews,
        samples_per_genre=args.samples_per_genre,
        train_count=args.train_count,
    )

    label2id, id2label = get_label_maps(train_labels)
    label_map_path = output_dir / 'label_maps.json'
    save_json({'label2id': label2id, 'id2label': id2label}, str(label_map_path))

    # defining tokenizer here to avoid loading the tokenizer before data preparation, which can save time if the data preparation step is the bottleneck.
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    train_encodings = encode_texts(tokenizer, train_texts, max_length=args.max_length)
    test_encodings = encode_texts(tokenizer, test_texts, max_length=args.max_length)

    train_labels_encoded = [label2id[label] for label in train_labels]
    test_labels_encoded = [label2id[label] for label in test_labels]

    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    # defining model here to avoid loading the model before data preparation, which can save time if the data preparation step is the bottleneck.
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(id2label),
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=args.logging_steps,
        eval_strategy='steps',
        eval_steps=args.logging_steps,
        save_strategy='epoch',
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        'model_name': args.model_name,
        'output_dir': str(output_dir),
        'num_labels': len(id2label),
        'label2id': label2id,
        'id2label': id2label,
    }
    save_json(metadata, str(output_dir / 'train_metadata.json'))
    print(f'Training complete. Model and tokenizer saved to {output_dir}')


if __name__ == '__main__':
    main()
