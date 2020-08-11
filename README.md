## Models

<p align="center"><img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>
models` is a tutorial for who is studying NLP(Natural Language Processing) using **TensorFlow** and **Pytorch**. 



## Curriculum - (Example Purpose)

#### 1. Basic Embedding Model

- 1-1. [NNLM(Neural Network Language Model)](https://github.com/graykode/nlp-tutorial/tree/master/1-1.NNLM) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  - Colab - [NNLM_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM_Tensor.ipynb), [NNLM_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM_Torch.ipynb)
- 1-2. [Word2Vec(Skip-gram)](https://github.com/graykode/nlp-tutorial/tree/master/1-2.Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec_Tensor(NCE_loss).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram_Tensor(NCE_loss).ipynb), [Word2Vec_Tensor(Softmax).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram_Tensor(Softmax).ipynb), [Word2Vec_Torch(Softmax).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram_Torch(Softmax).ipynb)
- 1-3. [FastText(Application Level)](https://github.com/graykode/nlp-tutorial/tree/master/1-3.FastText) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-3.FastText/FastText.ipynb)



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](https://github.com/graykode/nlp-tutorial/tree/master/2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - Colab - [TextCNN_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN_Tensor.ipynb), [TextCNN_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN_Torch.ipynb)
- 2-2. DCNN(Dynamic Convolutional Neural Network)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](https://github.com/graykode/nlp-tutorial/tree/master/3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN_Tensor.ipynb), [TextRNN_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN_Torch.ipynb)
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM_Tensor.ipynb), [TextLSTM_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM_Torch.ipynb)
- 3-3. [Bi-LSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**
  - Colab - [Bi_LSTM_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM_Tensor.ipynb), [Bi_LSTM_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM_Torch.ipynb)



#### 4. Attention Mechanism

- 4-1. [Seq2Seq](https://github.com/graykode/nlp-tutorial/tree/master/4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab - [Seq2Seq_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq_Tensor.ipynb), [Seq2Seq_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq_Torch.ipynb)
- 4-2. [Seq2Seq with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention)_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)_Tensor.ipynb), [Seq2Seq(Attention)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention)_Torch.ipynb)
- 4-3. [Bi-LSTM with Attention](https://github.com/graykode/nlp-tutorial/tree/master/4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**
  - Colab - [Bi_LSTM(Attention)_Tensor.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention)_Tensor.ipynb), [Bi_LSTM(Attention)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention)_Torch.ipynb)



#### 5. Model based on Transformer

- 5-1.  [The Transformer](https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - Colab - [Transformer_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer_Torch.ipynb), [Transformer(Greedy_decoder)_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder)_Torch.ipynb)
- 5-2. [BERT](https://github.com/graykode/nlp-tutorial/tree/master/5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT_Torch.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT_Torch.ipynb)

|           Model            |              Example               |   Framework   | Lines(torch/tensor) |
| :------------------------: | :--------------------------------: | :-----------: | :-----------------: |
|            NNLM            |         Predict Next Word          | Torch, Tensor |        67/83        |
|     Word2Vec(Softmax)      |   Embedding Words and Show Graph   | Torch, Tensor |        77/94        |
|          TextCNN           |      Sentence Classification       | Torch, Tensor |        94/99        |
|          TextRNN           |         Predict Next Step          | Torch, Tensor |        70/88        |
|          TextLSTM          |            Autocomplete            | Torch, Tensor |        73/78        |
|          Bi-LSTM           | Predict Next Word in Long Sentence | Torch, Tensor |        73/78        |
|          Seq2Seq           |            Change Word             | Torch, Tensor |       93/111        |
|   Seq2Seq with Attention   |             Translate              | Torch, Tensor |       108/118       |
|   Bi-LSTM with Attention   |  Binary Sentiment Classification   | Torch, Tensor |       92/104        |
|        Transformer         |             Translate              |     Torch     |        222/0        |
| Greedy Decoder Transformer |             Translate              |     Torch     |        246/0        |
|            BERT            |            how to train            |     Torch     |        242/0        |




## Dependencies

- Python 3.5+
- Tensorflow 1.12.0+
- Pytorch 1.2.0+
- Plan to add Keras Version



## Author

- Author Email : lyj157175@163.com
- GitHub: https://github.com/lyj157175/Models
