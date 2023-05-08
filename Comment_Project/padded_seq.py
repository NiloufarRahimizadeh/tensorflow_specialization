from tensorflow.keras.preprocessing.sequence import pad_sequences




def seq_and_pad(tokenizer, sentences, padding, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding=padding, maxlen=maxlen)
    return padded_sequences