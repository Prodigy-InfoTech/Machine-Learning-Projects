# Sentiment Analysis of Movie Reviews

## Overview

Welcome to the Sentiment Analysis of Movie Reviews project repository! This project focuses on employing a Bidirectional LSTM (Long Short-Term Memory) deep learning model to analyze the sentiment of movie reviews. The main components of this repository include:

- **sentiment_analysis_movie_reviews.ipynb**: This Jupyter Notebook contains the code for the sentiment analysis model. You can explore the code, run cells, and understand how the Bidirectional LSTM is trained and evaluated.

- **requirements.txt**: This file lists all the Python libraries and dependencies required to run the code in the Jupyter Notebook. Install these dependencies using the following command:

    ```
    pip install -r requirements.txt
    ```

- **IMDBDataset.csv**: This CSV file contains the IMDB dataset used for training and testing the sentiment analysis model. Review the dataset to understand its structure before running the notebook.

## Getting Started

To get started with the Sentiment Analysis of Movie Reviews project, follow these steps:

1. Clone the repository to your local machine:

    ```
    git clone https://github.com/Rc17git/Machine-Learning-Projects/tree/sentiment-analysis-movie
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook:

    ```
    jupyter notebook SA_movie_reviews.ipynb
    ```

4. Run the cells in the notebook to train and evaluate the sentiment analysis model.

## Dataset

The dataset (`IMDBataset.csv`) contains labeled examples of movie reviews, indicating their sentiment (positive or negative). It is crucial to understand the dataset structure to interpret the model's results accurately.

## Model Performance

The Bidirectional LSTM model achieved an impressive accuracy of 90% in sentiment classification. The evaluation metrics, including precision, recall, and F1 score, are presented in the notebook, providing insights into the model's performance.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request. Contributions are welcome!

