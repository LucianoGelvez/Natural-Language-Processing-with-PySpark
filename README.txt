# Spam Detection with PySpark: A Natural Language Processing Project

This project implements a spam detection filter using PySpark. It uses a dataset of text messages classified as either 'ham' (legitimate) or 'spam' and leverages various PySpark ML libraries to preprocess the text and train a logistic regression model for classification.

## Dataset

The dataset used in this project is a text file with messages and their labels.  The data set consists of volunteered text messages from a study in Singapore and some spam texts from a UK reporting site.

The dataset includes the following columns:

*   **class:**  The label of the message (either 'ham' or 'spam')
*   **text:** The content of the text message.

## Libraries

The following Python libraries are used in the project:

*   `pyspark.sql`: For data manipulation with Spark SQL.
*   `pyspark.sql.functions`: For custom SQL functions such as length of text.
*   `pyspark.ml.feature`: For feature engineering (Tokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler, StringIndexer).
*   `pyspark.ml.classification`: For the logistic regression model.
*   `pyspark.ml`: For the Pipeline.
*   `pyspark.ml.evaluation`: For evaluating the model.

## Process

1.  **Data Loading:** The dataset is loaded from a text file using Spark and the columns are renamed to `class` and `text`.
2.  **Feature Engineering:**
    *   A new feature `length` is added using the `length()` function on the `text` column.
    *   The `text` column is tokenized into words using `Tokenizer`.
    *   Stop words are removed using `StopWordsRemover`.
    *   The tokens are then vectorized with `CountVectorizer`.
    *   The Inverse Document Frequency `IDF` is applied to the vectorized text.
    *   The `tf_idf` vector and the `length` of the text are combined into a single feature vector `features`.
3.  **Data Preparation:**
    *   The `class` column is converted to numerical values using `StringIndexer` and this feature is named as label.
    *   A `Pipeline` is created to chain all the feature engineering stages.
    *   The data is split into train and test datasets.
4.  **Model Training:**
    *   A Logistic Regression model is initialized.
    *   The model is trained using the training dataset and the features generated previously.
5.  **Model Evaluation:**
    *   The trained model is applied to the test dataset to generate predictions
    *   The model's performance is evaluated using `MulticlassClassificationEvaluator` (calculating accuracy) on the test dataset.

## Usage

To run this code, make sure you have the following installed:

*   Python 3.6+
*   Apache Spark
*   PySpark

You can run the code directly in a Spark environment, using a Jupyter notebook or a Python script.

## Additional Considerations
* This project implements supervised learning, therefore there is a label to evaluate the model.
* PySpark's DAGScheduler messages can be ignored, but should be watched for optimization purposes.