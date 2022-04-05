# Classification Challenge
This repo contains my solution for the task described below. The predictions are stored within `purchase_probabilities.csv`. Reasoning, analysis and explaniations can be found in `src/purchase_probabilities.ipynb` and mentioned notebooks. Most action is happening behind the scenes inside `src/utils.py`. Outtakes are available, too.

<br>
<br>

## Task
###  Instructions
You are provided with the following data:

a history of customer purchases in a store for calendar weeks 1 to 49 of the same year for
2000 customers and an assortment of 100 products.

Predict the purchase probability for each customer product pair in week 50.

### Requirements
The solution should be done in Python. Please provide the details of your analysis in a Jupyter
Notebook together with a **csv file containing purchase probability predictions** in the format consumer,
product, probability.

### Data
Train_data.csv contains the following columns:
- `consumer_id` -- unique id of a consumer
- `product_id` -- unique id of a product
- `week` -- calendar week number
- `price` -- price paid by the consumer