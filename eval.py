import argparse
from pathlib import Path

from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer

from data import GENRE_URL_DICT, build_genre_reviews, encode_texts, load_pickle, save_pickle, split_train_test
from utils import MyDataset, compute_metrics, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a saved BERT classifier on Goodreads genre review test data.')
    parser.add_argument('--dataset_pickle', default='genre_reviews_dict.pickle')
    parser.add_argument('--model_dir', default='./results')
    parser.add_argument('--output_dir', default='./evaluation')
    parser.add_argument('--samples_per_genre', type=int, default=1000)
    parser.add_argument('--train_count', type=int, default=800)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--force_download', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    label_map_path = model_dir / 'label_maps.json'
    if not label_map_path.exists():
        raise FileNotFoundError(f'Label map file not found: {label_map_path}')

    label_maps = load_json(str(label_map_path))
    label2id = {str(key): int(value) for key, value in label_maps.get('label2id', {}).items()}
    id2label = {int(key): value for key, value in label_maps.get('id2label', {}).items()}

    pickle_path = Path(args.dataset_pickle)
    if not pickle_path.exists() and args.force_download:
        genre_reviews = build_genre_reviews(genre_url_dict=GENRE_URL_DICT)
        save_pickle(genre_reviews, pickle_path)

    if not pickle_path.exists():
        raise FileNotFoundError(f'Dataset pickle not found: {pickle_path}')

    genre_reviews = load_pickle(pickle_path)
    _, _, test_texts, test_labels = split_train_test(
        genre_reviews_dict=genre_reviews,
        samples_per_genre=args.samples_per_genre,
        train_count=args.train_count,
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    test_encodings = encode_texts(tokenizer, test_texts, max_length=args.max_length)

    test_labels_encoded = [label2id[label] for label in test_labels]
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(id2label))

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    result = trainer.predict(test_dataset)

    predictions = result.predictions.argmax(-1).tolist()
    predicted_labels = [id2label[pred] for pred in predictions]

    report = classification_report(test_labels, predicted_labels, digits=4)
    evaluation_results = {
        'metrics': result.metrics,
        'classification_report': report,
    }

    save_json(evaluation_results, str(output_dir / 'evaluation_results.json'))
    with open(str(output_dir / 'classification_report.txt'), 'w', encoding='utf-8') as handle:
        handle.write(report)

    print('Evaluation complete.')
    print(report)
    print(f'Evaluation results saved to {output_dir}')


if __name__ == '__main__':
    main()
