# AI-Poet

Try to train a neural network to automatically write Chinese poems

## Usage

### Training

You can select the model (CharRNN / Seq2Seq) by changing `config.py`

To train the model, just input the following command in the terminal

```shell script
python main.py
```

### Playing(×

~~Though so-called playing, it's not fun at all!~~

But you can still try it by typing the following command

```shell script
python write_poem.py
```

### References

1. [LSTM_poem](https://github.com/braveryCHR/LSTM_poem)
2. [CharRNN](https://github.com/chenyuntc/pytorch-book/tree/master/chapter9-神经网络写诗(CharRNN))
3. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
4. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078v3)
5. [IBM/pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)
