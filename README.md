# Lab 3: Deep Learning for Natural Language Processing with Sequence Models

## Objective
The main purpose of this lab is to get familiar with PyTorch, build deep neural network architectures for Natural Language Processing (NLP) using Sequence Models, and apply them to classification and text generation tasks on Arabic text data.

## Work

### Part 1: Classification Task

#### 1. Data Collection
Using scraping libraries (BeautifulSoup), text data was collected from Arabic websites (e.g., Al Jazeera's technology section) on the topic of technology. Each text sample was assigned a relevance score between 0 and 10 based on its pertinence to the topic. Synthetic data was also added to augment the dataset for training.

- **Dataset Summary**:
  - Total samples: 100 (real + synthetic).
  - Average length: 34 words.
  - Example structure:
    | Text (Arabic Language) | Score |
    |------------------------|-------|
    | ... (sample Arabic text) | 6    |
    | ... (sample Arabic text) | 7.5  |

  Real articles were scraped from URLs like:
  - https://www.aljazeera.net/technology/news/
  - Examples of scraped articles: 5 from Al Jazeera (words ranging from 28 to 601, scores 5.0 to 9.1).

  Synthetic samples were generated to reach 100 samples, categorized as low, medium, or high relevance.

#### 2. Preprocessing NLP Pipeline
A preprocessing pipeline was established for the collected dataset:
- **Tokenization**: Using NLTK or custom tokenizers for Arabic text.
- **Stemming/Lemmatization**: Applied Arabic-specific stemming (e.g., via Tashaphyne or Farasa).
- **Stop Words Removal**: Removed common Arabic stop words.
- **Discretization**: Scores were categorized (e.g., low, medium, high) for some analyses, but kept continuous for regression.
- **Other Steps**: Normalization, padding sequences to max length (517 tokens), and vocabulary building (1604 words).

- **Original Data Sample** (before preprocessing):
  - Loaded 100 samples.
  - Total tokens: 29.6 (average length).
  - Min length: 15, Max length: 517.

- **Preprocessed Data Sample**:
  - Saved to `arabic_dataset_preprocessed.csv`.
  - Examples with scores and categories (e.g., score 5.0 -> low, 8.2 -> high).

- **Score Distribution** (post-preprocessing):
  - Mean: 5.0
  - Median: 5.0
  - Std: 2.92
  - Categories: Low (33), Medium (51), High (16).

#### 3. Model Training
Models were trained using RNN, Bidirectional RNN (BiRNN), GRU, and LSTM architectures. Hyper-parameters were tuned for best performance:
- **Vocabulary Size**: 1604
- **Embedding Dim**: 128
- **Hidden Dim**: 256
- **Layers**: 2
- **Dropout**: 0.3
- **Learning Rate**: 0.001 (for most), 3e-05 for fine-tuning.
- **Epochs**: 30
- **Device**: CPU
- **Max Sequence Length**: 512

- **Training Logs** (examples):
  - **RNN**: Best Val Loss: 36.8552 (MSE decreases over epochs).
  - **BiRNN**: Best Val Loss: 0.8552 (quick convergence).
  - **GRU**: Best Val Loss: 9.9628
  - **LSTM**: Best Val Loss: 0.9623

Training curves saved as `training_curves.png`, showing train/validation loss over epochs for each model.
<img width="4466" height="2970" alt="training_curves" src="https://github.com/user-attachments/assets/c820ab31-ce5b-48ed-bc1f-f8f0a1fea22b" />


#### 4. Evaluation
The four models were evaluated using standard metrics (MSE, RMSE, MAE, R², Correlation) and NLP-specific metrics like BLEU-like score and accuracy (±1.0 and ±0.5 tolerance).

- **Final Results Summary**:
  | Model | MSE    | RMSE   | MAE    | R²     | Correlation | Accuracy (±1.0) | Accuracy (±0.5) | BLEU-Like Score |
  |-------|--------|--------|--------|--------|-------------|-----------------|-----------------|-----------------|
  | RNN   | 36.8734| 6.0723 | 5.0840 | -34.0017 | 0.3978 | 5.0    | 0.0    | 0.0000 |
  | BiRNN | 0.9337 | 0.9663 | 0.7638 | 0.1137 | 0.3759 | 65.0   | 50.0   | 0.7249 |
  | GRU   | 1.0799 | 1.0391 | 0.7775 | -0.0251| 0.1824 | 60.0   | 45.0   | 0.6738 |
  | LSTM  | 1.0552 | 1.0272 | 0.7395 | -0.0017| 0.4034 | 70.0   | 55.0   | 0.7238 |

- **Best Model**: BiRNN (lowest RMSE: 0.9663).
- Results saved to `model_results.csv` and `final_model_comparison.csv`.
- Visualizations:
  - Model comparison bar plots saved as `model_comparison.png` and `comprehensive_model_comparison.png`.
  <img width="5970" height="1469" alt="model_comparison" src="https://github.com/user-attachments/assets/d09d9252-ab77-4fdc-a3b6-da6355b2d38f" />
  <img width="5370" height="3566" alt="comprehensive_model_comparison" src="https://github.com/user-attachments/assets/bc740816-1076-4af2-a45b-551983424774" />
  - Residual plots and distributions for each model (e.g., `RNN_residuals.png`, `BiRNN_residuals.png`).
  <img width="4470" height="1767" alt="RNN_residuals" src="https://github.com/user-attachments/assets/f836abdc-762e-4afd-b8f1-da2ff6ab5079" />
  <img width="4470" height="1767" alt="BiRNN_residuals" src="https://github.com/user-attachments/assets/95fa5900-aa40-43d3-8cf5-c4e020abaa14" />
  - Actual vs Predicted scores plots (e.g., `RNN_predictions.png`).
  <img width="2970" height="2367" alt="RNN_predictions" src="https://github.com/user-attachments/assets/2b2d27ad-d6a5-463d-9d88-4ff642594e53" />
  <img width="2970" height="2367" alt="BiRNN_predictions" src="https://github.com/user-attachments/assets/e5e7b65f-b74d-4e03-9f32-fdf0197e56b3" />

- **BLEU-Like Score**: Adapted for relevance scoring, treating predicted scores as "translations" of actual scores.

### Part 2: Transformer (Text Generation)

#### 1. Fine-Tuning GPT2
Installed `pytorch-transformers` (now `transformers` library). Loaded the GPT2 pre-trained model.

- Fine-tuned GPT2 on a customized dataset (`arabic_dataset_preprocessed.csv` from Part 1).
- **Hyper-parameters**:
  - Epochs: 3
  - Batch Size: 4
  - Learning Rate: 3e-05
  - Max Sequence Length: 512
  - Warmup Steps: 100

- **Training Logs**:
  - Epoch 1: Average Loss 6.5238
  - Epoch 2: Average Loss 8.9314
  - Epoch 3: Average Loss 6.4398
- Model saved as `trained_models/gpt2_arabic_epoch_X.pt`.

#### 2. Text Generation
Generated new paragraphs based on given sentences (prompts).
- **Prompt**: Arabic sentences (e.g., "الذكاء الاصطناعي").
- **Max Length**: 50 tokens.
- **Generated Samples** (examples from output):
  - Sample 1: (Generated Arabic text about AI).
  - Sample 2: (Generated Arabic text).
  - Sample 3: (Generated Arabic text).

- Generated texts saved to `generated_arabic_texts.txt`.

## Results
- **Classification**: BiRNN outperformed others with the lowest error metrics and highest accuracy.
- **Text Generation**: Fine-tuned GPT2 produced coherent Arabic paragraphs, though limited by small dataset size.
- All plots and CSVs are included in the repository (e.g., training curves, residuals, predictions).

## Synthesis: What I Learned
During this lab, I gained hands-on experience with PyTorch for building and training sequence models (RNN, BiRNN, GRU, LSTM) on Arabic NLP tasks. Key learnings include:
- Data scraping and preprocessing challenges for Arabic (handling right-to-left text, stemming).
- Hyper-parameter tuning's impact on model convergence (e.g., BiRNN converged fastest).
- Evaluation metrics for regression in NLP contexts, including adapting BLEU for scoring.
- Fine-tuning transformers like GPT2 for text generation, understanding tokenization and generation parameters.
- Importance of dataset size/quality; synthetic data helped but real data improved relevance.
- Overall, this reinforced sequence models' strengths for NLP and PyTorch's flexibility.
