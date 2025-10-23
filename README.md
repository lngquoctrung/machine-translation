# 🌐 English-Vietnamese Machine Translation

BiLSTM + Multi-Head Attention model cho neural machine translation với tất cả optimization techniques.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎯 **High Accuracy**: BiLSTM + Multi-Head Attention với BLEU score ~25-30
- ⚡ **GPU Optimized**: Fit trong 16GB GPU, train < 10 giờ
- 🔍 **Advanced Techniques**: 
  - Label Smoothing (α=0.1)
  - Layer Normalization
  - Residual Connections
  - Warmup + Cosine LR Schedule
  - Mixed Precision Training (FP16)
  - Beam Search Decoding (k=5)
- 🌐 **Web Interface**: Gradio app với UI đẹp
- 📊 **BLEU Evaluation**: Automatic quality assessment
- 📈 **Training Monitoring**: TensorBoard integration

## 🏗️ Architecture

Input (English)
↓
Embedding Layer (64 dim)
↓
BiLSTM Encoder (128 units × 2)
↓
Layer Normalization
↓
Multi-Head Attention (2 heads)
↓
Residual Connection (Add & Norm)
↓
LSTM Decoder (256 units)
↓
Layer Normalization
↓
Dense Layer (Softmax)
↓
Output (Vietnamese)

text

### Key Techniques

| Technique | Purpose | Impact |
|-----------|---------|--------|
| **Mixed Precision** | Memory efficiency | -40% GPU memory, 2-3x faster |
| **Label Smoothing** | Generalization | +1-2 BLEU points |
| **Layer Norm** | Training stability | +0.5-1 BLEU points |
| **Beam Search** | Better inference | +3-5 BLEU points |
| **LR Scheduling** | Optimal convergence | +1-2 BLEU points |

## 📁 Project Structure

```bash
machine-translation/
├── config/
│ ├── init.py
│ ├── model_config.py # Model configuration
│ └── train_config.py # Training hyperparameters
│
├── src/
│ ├── data/
│ │ ├── preprocessing.py # Data preprocessing
│ │ └── dataset.py # Dataset utilities
│ ├── models/
│ │ ├── bilstm_attention.py # BiLSTM + Attention
│ │ └── lstm_attention.py # LSTM + Attention
│ ├── training/
│ │ ├── trainer.py # Training logic
│ │ ├── loss_functions.py # Label smoothing
│ │ ├── schedulers.py # LR schedules
│ │ └── callbacks.py # Custom callbacks
│ ├── evaluation/
│ │ ├── metrics.py # BLEU score
│ │ └── beam_search.py # Beam search decoder
│ └── utils/
│ ├── gpu_utils.py # GPU memory manager
│ └── helpers.py # Helper functions
│
├── app/
│ └── gradio_app.py # Gradio web interface
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_model_training.ipynb
│ └── 03_evaluation.ipynb
│
├── data/
│ ├── raw/ # Original datasets
│ └── processed/ # Processed data
│
├── models/
│ ├── checkpoints/ # Training checkpoints
│ ├── saved_models/ # Final models
│ └── tokenizers/ # Saved tokenizers
│
├── logs/ # Training logs
│
├── main.py # Training script
├── requirements.txt # Dependencies
├── README.md # This file
└── .gitignore # Git ignore

```

## 🚀 Quick Start

### Installation

Clone repository (hoặc download)
cd machine-translation

Create virtual environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Create directories
mkdir -p data/raw data/processed
mkdir -p models/{checkpoints,saved_models,tokenizers}
mkdir -p logs

text

### Prepare Data

Copy your datasets
cp /path/to/dataset_en.txt data/raw/
cp /path/to/dataset_vi.txt data/raw/

text

**Data format:**
- `dataset_en.txt`: 1 English sentence per line
- `dataset_vi.txt`: 1 Vietnamese sentence per line (corresponding)

### Training

#### Train BiLSTM Model (Recommended)

Train BiLSTM với tất cả optimizations
python main.py --model bilstm

Monitor với TensorBoard
tensorboard --logdir logs/

text

#### Train LSTM Model (For comparison)

Train LSTM (uni-directional)
python main.py --model lstm

text

**Expected training time:**
- BiLSTM: ~5-7 hours (GPU 16GB)
- LSTM: ~4-6 hours (GPU 16GB)

### Run Gradio Demo

Launch web interface
python app/gradio_app.py

Access at http://localhost:7860
text

**Features:**
- Real-time translation
- Beam search vs Greedy decoding comparison
- BLEU score calculation (with reference)
- Translation history
- Beautiful UI

## 📊 Model Performance

### Training Results

| Model | Parameters | GPU Memory | Training Time | Val Loss | Val Accuracy |
|-------|-----------|-----------|--------------|----------|-------------|
| **BiLSTM** | ~15-20M | 6-8 GB | 5-7h | ~2.5 | ~69% |
| **LSTM** | ~10-15M | 4-6 GB | 4-6h | ~2.8 | ~65% |

### Translation Quality

| Model | BLEU Score | Inference Time |
|-------|-----------|----------------|
| **BiLSTM + Beam** | **~25-30** | 0.2-0.4s |
| BiLSTM + Greedy | ~20-25 | 0.1-0.2s |
| LSTM + Beam | ~22-27 | 0.15-0.3s |
| LSTM + Greedy | ~18-23 | 0.08-0.15s |

## 🛠️ Configuration

Edit `config/model_config.py` to customize:

class ModelConfig:
# Vocabulary
MAX_VOCAB_SIZE_EN = 25000 # English vocab size
MAX_VOCAB_SIZE_VI = 20000 # Vietnamese vocab size
MIN_WORD_FREQUENCY = 2 # Filter rare words

text
# Architecture
EMBEDDING_DIM = 64         # Embedding dimension
LSTM_UNITS = 128           # LSTM hidden units
ATTENTION_HEADS = 2        # Multi-head attention heads

# Training
BATCH_SIZE = 256           # Batch size
EPOCHS = 100               # Training epochs
LEARNING_RATE = 0.001      # Peak learning rate

# Optimization
USE_MIXED_PRECISION = True # Enable FP16
GPU_MEMORY_LIMIT = 15000   # GPU memory limit (MB)
LABEL_SMOOTHING = 0.1      # Label smoothing factor

# Inference
BEAM_WIDTH = 5             # Beam search width
text

## 📈 Model Comparison

### BiLSTM vs LSTM

**BiLSTM Advantages:**
- ✅ Better accuracy (+3-5 BLEU)
- ✅ Captures bidirectional context
- ✅ Better for complex sentences

**BiLSTM Disadvantages:**
- ❌ More parameters (~50% more)
- ❌ Slower training (~20-30% slower)
- ❌ More GPU memory (~30-40% more)

**LSTM Advantages:**
- ✅ Faster training
- ✅ Less memory
- ✅ Good for real-time applications

**Recommendation:** Use **BiLSTM** for best quality, **LSTM** for speed.

## 🎯 Usage Examples

### Training

Train BiLSTM (recommended)
python main.py --model bilstm

Train LSTM (faster)
python main.py --model lstm

text

### Gradio App

Launch demo
python app/gradio_app.py

Custom port
python app/gradio_app.py --port 8080

text

### Programmatic Usage

import tensorflow as tf
import pickle
from src.evaluation.beam_search import BeamSearchDecoder

Load model
model = tf.keras.models.load_model('models/saved_models/bilstm_model.h5')

Load tokenizers
with open('models/tokenizers/tokenizer_en.pkl', 'rb') as f:
tokenizer_en = pickle.load(f)
with open('models/tokenizers/tokenizer_vi.pkl', 'rb') as f:
tokenizer_vi = pickle.load(f)

Create decoder
decoder = BeamSearchDecoder(model, tokenizer_en, tokenizer_vi, 40, 50, beam_width=5)

Translate with beam search
translation, candidates = decoder.decode_beam_search(
decoder.preprocess("Hello, how are you?")
)
print(f"Translation: {translation}")
print(f"Alternatives: {candidates}")

text

## 🔧 Troubleshooting

### GPU Memory Issues

Edit config/model_config.py
GPU_MEMORY_LIMIT = 12000 # Giảm xuống 12GB
BATCH_SIZE = 128 # Giảm batch size

text

### Training Too Slow

Enable mixed precision
USE_MIXED_PRECISION = True

Increase batch size
BATCH_SIZE = 512

text

### Low Accuracy

Increase model capacity
LSTM_UNITS = 256
EMBEDDING_DIM = 128

Increase vocabulary
MAX_VOCAB_SIZE_EN = 30000
MAX_VOCAB_SIZE_VI = 25000

More epochs
EPOCHS = 150

text

## 📚 Notebooks

Explore analysis notebooks in `notebooks/`:

1. **`01_data_exploration.ipynb`**: Data analysis, vocabulary distribution
2. **`02_model_training.ipynb`**: Training both models, comparison
3. **`03_evaluation.ipynb`**: BLEU score evaluation, translation tests

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open pull request

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- TensorFlow & Keras teams
- Gradio for amazing UI framework
- Research papers on attention mechanisms and beam search

## 📧 Contact

- GitHub Issues: [Open an issue](https://github.com/yourusername/machine-translation/issues)
- Email: your.email@example.com

---

**Built with ❤️ using TensorFlow, Keras & Gradio**

**Keywords:** Machine Translation, Neural Machine Translation, BiLSTM, LSTM, Attention Mechanism, Seq2Seq, English-Vietnamese, Deep Learning, NLP, TensorFlow