# Distill Bert Classification for Books

## Run summary:
| Metrics | Final Value |
|----------|--------|
| `accuracy` | 0.595 |
| `F1` | 0.59263 |
| `loss` | 1.16694 |

> GC training file - https://colab.research.google.com/drive/1YAwEBD3HNVf3pRIlvzMbzlUdk0kDIOfw#scrollTo=sZFvYxowugsE
> 
> WandB dashboard - https://wandb.ai/g25ait2048-iit-jodhpur/mlops-assignment2/workspace?nw=nwuserg25ait2048

## Setup and Training
> Model training done on local and collab,
> local env - create and activate using UV virt env
- python3 train.py
- python3 eval.py
- uv pip install -r requirements.txt
- uv pip install dotenv
- python3 train_with_wandb.py
- python3 eval_with_wandb.py

## Evaluation Summary
| Genre | precision | recall | f1-score | support|
|----------|--------|----------|--------|---------|
|children  | 0.7181  |  0.6750 |   0.6959 |      200|
|comics_graphic| 0.8138 |0.7650  |  0.7887      | 200|
  |  fantasy_paranormal |    0.4340   | 0.5100   | 0.4690     |  200|
   |  history_biography  |   0.5751  |  0.5550|0.5649   |    200|
|mystery_thriller_crime   |  0.5931 |  0.6050 |   0.5990   |    200|
 |               poetry    | 0.7617 |   0.8150  |  0.7874  |     200|
  |             romance    | 0.6119 |   0.6150   | 0.6135 |      200|
   |        young_adult    | 0.4011|    0.3550    |0.3767|       200|

              accuracy                         0.6119      1600
             macro avg     0.6136    0.6119    0.6119      1600
          weighted avg     0.6136    0.6119    0.6119      1600
