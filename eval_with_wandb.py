import argparse
import json
from pathlib import Path

import wandb
from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer

from data import GENRE_URL_DICT, build_genre_reviews, encode_texts, load_pickle, save_pickle, split_train_test
from utils import MyDataset, compute_metrics, load_json, save_json
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.setdefault('WANDB_API_KEY', os.getenv('WANDB_API_KEY'))

# Note: Uncomment the below line if running on collab and set the WANDB_API_KEY in your Colab environment variables.
# Also comment line 12 and line 13 if you want to use the API key from Colab environment variables.
# from google.colab import userdata
# os.environ["WANDB_API_KEY"] = userdata.get('WANDB_API_KEY') #IMPORTANT: Set your W&B API key in Colab environment variables for this to work.


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a saved BERT classifier and log metrics to Weights & Biases.')
    parser.add_argument('--dataset_pickle', default='genre_reviews_dict.pickle')
    parser.add_argument('--model_dir', default='./results_wandb')
    parser.add_argument('--output_dir', default='./evaluation_wandb')
    parser.add_argument('--project_name', default='mlops-assignment2')
    parser.add_argument('--run_name', default='distilbert-eval-run')
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

    label2id = {value: key for key, value in id2label.items()}
    test_labels_encoded = [label2id[label] for label in test_labels]

    test_dataset = MyDataset(test_encodings, test_labels_encoded)
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir), num_labels=len(id2label))

    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config={
            'model_dir': str(model_dir),
            'dataset_pickle': str(pickle_path),
            'samples_per_genre': args.samples_per_genre,
            'train_count': args.train_count,
            'max_length': args.max_length,
        },
    )

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print('Evaluation results:', eval_results)

    preds = trainer.predict(test_dataset).predictions.argmax(-1)
    labels = [item['labels'].item() for item in test_dataset]
    report = classification_report(labels, preds, target_names=[id2label[i] for i in sorted(id2label)], output_dict=True)

    report_path = output_dir / 'eval_report.json'
    with report_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)

    final_accuracy = eval_results.get('eval_accuracy')
    final_loss = eval_results.get('eval_loss')
    final_f1 = report.get('weighted avg', {}).get('f1-score')

    wandb.log({
        'final/loss': final_loss,
        'final/accuracy': final_accuracy,
        'final/f1': final_f1,
    })

    artifact = wandb.Artifact('eval-report', type='evaluation')
    artifact.add_file(str(report_path))
    wandb.log_artifact(artifact)
    wandb.finish()

    evaluation_results = {
        'metrics': eval_results,
        'classification_report': report,
    }
    save_json(evaluation_results, str(output_dir / 'evaluation_results.json'))

    print(f'Evaluation complete. Report saved to {report_path}')


if __name__ == '__main__':
    main()
