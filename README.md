# Real-Time Credit Card Fraud Detection

## Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning. It includes a Jupyter Notebook for the project workflow and a Streamlit application for demonstrating fraud detection.

## Project Structure

```text
Real-Time-Fraud-Detection/
│
├── app.py
├── Fraud Detection.ipynb
├── .gitignore
├── README.md
│
├── models/
│   └── model.pkl
│
└── data/
    └── demo_stream.csv
```

## Technologies Used

- Python
- Machine Learning
- Pandas
- Streamlit
- Joblib
- Jupyter Notebook

## Application Features

- Machine learning-based fraud prediction
- Transaction scoring using prediction probabilities
- Streamlit-based application interface
- Demonstration using transaction stream data

## Installation and Setup

1. Clone this repository:

```bash
git clone https://github.com/VaishnaviBallolli/Real-Time-Fraud-Detection.git
```

2. Navigate to the project directory:

```bash
cd Real-Time-Fraud-Detection
```

3. Install the required libraries:

```bash
pip install streamlit pandas joblib
```

4. Make sure the required model and demonstration data files are available:

- `models/model.pkl`
- `data/demo_stream.csv`

5. Run the Streamlit application:

```bash
streamlit run app.py
```

## Dataset

The project uses credit card transaction data for fraud detection.

The original dataset is excluded from this public repository because of its size. Obtain the dataset separately and place it in the appropriate project directory when running the notebook.

## Author

**Vaishnavi Ballolli**

GitHub: [VaishnaviBallolli](https://github.com/VaishnaviBallolli)
