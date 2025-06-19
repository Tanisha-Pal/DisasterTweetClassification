#  Tweet Disaster Detection



##  Introduction

Welcome to the **Tweet Disaster Detection** repository! This project is an advanced Natural Language Processing (NLP) solution designed to identify disaster-related tweets in real-time. By leveraging cutting-edge machine learning and deep learning techniques, this system empowers decision-makers with timely information to respond effectively to emergencies. 🌟

With the explosion of social media usage, the ability to rapidly detect disaster events through user-generated content has become critical. Our solution is optimized for accuracy and reliability, ensuring robust disaster identification.

---

## 🌟 Key Features

- **State-of-the-Art Models**: Fine-tuned **BERT** transformer for high-precision tweet classification.
- **Real-Time Analysis**: Designed to process and classify tweets quickly and accurately.
- **Actionable Insights**: Focused on real-world applications, such as early disaster warnings and accurate reporting.
- **Scalable Solution**: Easily adaptable to different datasets or NLP tasks.

---

## 🔧 Libraries and Frameworks

This project utilizes several powerful tools:

- **[TensorFlow](https://www.tensorflow.org/)** and **[Keras](https://keras.io/)**: Core frameworks for implementing and fine-tuning the BERT model.
- **[Huggingface Transformers](https://huggingface.co/transformers/)**: Pre-trained BERT models and tokenization utilities for NLP tasks.
- **[scikit-learn](https://scikit-learn.org/)**: For traditional ML tasks like Naive Bayes classification and evaluation metrics.
- **[Matplotlib](https://matplotlib.org/)**: Visualization tools for model performance analysis.
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation and preprocessing for tweet analysis.

---

## 💡 Project Overview

In a flood of tweets generated every second, discerning disaster-related content is challenging. This system addresses this challenge by distinguishing tweets that indicate real disasters from irrelevant content, using a fine-tuned **BERT** model for exceptional performance.

### 🧠 Model Overview

Our primary model is a fine-tuned **BERT** transformer with the following pipeline:

1. **Preprocessing**:
   - Tweets are tokenized with BERT's tokenizer, converting text into token IDs, attention masks, and segment IDs.

2. **Model Architecture**:
   - A dense layer is added to the pre-trained BERT model to classify tweets as disaster-related or not.

   ```python
   input_word_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_word_ids')
   input_mask = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_mask')
   segment_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='segment_ids')

   pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
   clf_output = sequence_output[:, 0, :]
   out = Dense(1, activation='sigmoid')(clf_output)
   model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
   ```

3. **Training**:
   - Trained using **SGD optimizer** with learning rate `0.0001` and momentum `0.8`.
   - Metrics tracked: accuracy, precision, recall, and F1-score.

---

## 🚀 Results

| Model          | Precision | Recall | Accuracy | F1-Score |
|----------------|:---------:|:------:|:--------:|:--------:|
| **BERT**       | 86%       | 84%    | 85%      | 86%      |
| **Naive Bayes**| 82%       | 70%    | 56%      | 75%      |

### 📊 Visualizations

- **Learning Curves**: Visualize accuracy, precision, and recall across epochs.
- **Confusion Matrix**: Detailed analysis of model predictions.

---

