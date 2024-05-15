# Travel Insurance Recommendation AI System

![alt text](assest/bg.svg)

This repository contains the implementation of an AI system designed to predict flight delays and recommend travel insurance to customers based on their potential purchase intentions.

## Overview

Our AI system, Insurance-GPT, leverages a Large Language Model (LLM) to analyze customer interactions in real-time and predict flight delays using a deep learning model. This system provides personalized insurance recommendations to improve user experience and offers valuable insights for insurance pricing strategies.

## Features

- **Real-time Interaction**: Insurance-GPT interacts with users to understand their needs and preferences.
- **Flight Delay Prediction**: Utilizes a deep learning model to predict flight delays accurately.
- **Personalized Insurance Recommendations**: Provides tailored insurance suggestions based on predicted delays and customer sentiment.

## Datasets

We used three main datasets to train and evaluate our models:

1. **[Travel Insurance Dataset](https://www.kaggle.com/datasets/marwandiab/travel-insurance-dataset)**: Contains customer profiles and their potential to purchase insurance.
2. **[Twitter US Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment?resource=download&select=Tweets.csv)**: Provides user reviews about US airlines, categorized as positive or negative.
3. **[InsuranceQA-v2 Dataset](https://huggingface.co/datasets/soulhq-ai/insuranceQA-v2)**: Contains insurance-related questions and answers for training our model.

## Model Implementation

### LLM Fine-Tuning

1. **Supervised Fine-Tuning (SFT)**: On the insuranceQA-v2 dataset to improve domain-specific understanding.
2. **Odds Ratio Preference Optimization (ORPO)**: To enhance the model's generalization and alignment with human preferences.
3. **LoRA Fine-Tuning**: On the Travel Insurance Dataset to improve precision in predicting user purchase intent.

### Deep Learning Model

Our deep learning model uses the ASTGCN framework to predict flight delays based on spatio-temporal data. This framework captures both temporal and spatial correlations in the data for accurate predictions.

## Citation

If you use this code or dataset in your research, please cite our report using the following BibTeX entry:

```bibtex
@techreport{yang2024insurancegpt,
  author       = {Yuzhe Yang and Haoqi Zhang and Zhidong Peng and Yilin Guo and Tianji Zhou},
  title        = {Travel Insurance Recommendation AI System Based on Flight Delay Predictions and Customer Sentiment},
  institution  = {The Chinese University of Hong Kong, Shenzhen},
  year         = {2024},
  url          = {https://github.com/TobyYang7/Travel-Insurance-Recommendation-AI-System},
  note         = {Available at: \url{https://github.com/TobyYang7/Travel-Insurance-Recommendation-AI-System}}
}
```
