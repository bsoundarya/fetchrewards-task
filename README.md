# Multi-Task BERT Model for Sentiment Analysis and Classification

This repository contains implementations of a multi-task BERT model for sentiment analysis and classification. The model is capable of performing two tasks simultaneously: predicting the sentiment of a movie review (positive, negative, or neutral) and classifying the sentence into predefined numerical categories.

### Contents

- `fetch_task_1.ipynb`: A single-task BERT model for sentiment analysis.
- `fetch_task_2.ipynb`: A multi-task BERT model for sentiment analysis and classification with shared parameters.
- `fetch_task_4.ipynb`: An enhanced multi-task BERT model with separate learning rates for BERT parameters and task-specific parameters.

### Requirements

- Python 3.7+
- PyTorch 1.7+
- transformers 4.0+
- scikit-learn 0.24+
- numpy 1.19+

### Installation
- Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### Single-Task BERT (fetch_task_1.ipynb)

This script trains a single-task BERT model for sentiment analysis. The model is trained on a small sample dataset and can predict the sentiment of test sentences.

To run the script:
```bash
python fetch_task_1.ipynb
```

#### Multi-Task BERT with Shared Parameters (fetch_task_2.ipynb)

This script trains a multi-task BERT model with shared parameters for both sentiment analysis and classification tasks. The model is trained on a small sample dataset and can predict both the sentiment and category of test sentences.

To run the script:
```bash
python fetch_task_2.ipynb
```

#### Enhanced Multi-Task BERT with Separate Learning Rates (fetch_task_4.ipynb)

This script trains an enhanced multi-task BERT model with separate learning rates for BERT parameters and task-specific parameters. This setup allows for better fine-tuning and improved performance. The model is trained on a small sample dataset and can predict both the sentiment and category of test sentences.

To run the script:
```bash
python fetch_task_4.ipynb
```

### Training and Evaluation
#### The training loop for each script involves the following steps:
- Data Preparation: Tokenizing input sentences and encoding labels.
- Model Initialization: Initializing the BERT model and classifiers.
- Optimizer and Scheduler: Setting up the AdamW optimizer and learning rate scheduler.
- Training Loop: Training the model for a specified number of epochs, computing loss and updating model weights.
- Evaluation: Making predictions on test sentences and printing the results.

### Customization
You can customize the scripts by modifying the sample sentences, labels, hyperparameters and other settings as needed. The provided code serves as a template and can be extended for larger datasets and more complex tasks.

### License
This project is licensed under the MIT License. 

### Acknowledgements
This repository leverages the Hugging Face transformers library for BERT model implementation and tokenization. Special thanks to the open-source community for providing valuable resources and tools.

