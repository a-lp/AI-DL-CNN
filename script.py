from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, AveragePooling2D, Softmax, InputLayer
import csv
import logging
from numpy import argmax, array
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.math import confusion_matrix
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
print("Caricamento dipendenze")

# Variabili di configurazione
path = "data/"
model_path = "model/"
kmer_size = 3   # Dimensione k-mer per la suddivisione delle sequenze
slide_step = 1
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)
epochs = 200
batch_size = 256
# Variabili dati
sequence_words = set()  # K-mers
types = []  # Tipi delle sequenze: EI, IE, N
sequences = []  # Sequenze DNA
word_dictionary = {}
word_ohe = {}  # Mapping k-mers -> one hot encode
logging.basicConfig(filename="log",
                    filemode='a',
                    format='%(message)s\n',
                    level=logging.DEBUG)


def lettura_file(filename):
    ''' Lettura del file, con nome passato a parametro. Il metodo popola le variabili types, sequences e sequence_words, contenenti rispettivamente i tipi [EI, IE, N], le sequenze di DNA e i k-mers. '''
    print("Lettura del file di input")
    # Lettura del modello
    with open(("%s/%s") % (path, filename), "r") as f:
        csvreader = csv.reader(f, delimiter=',')
        for type_, name, sequence in csvreader:
            types.append(type_)
            sequences.append(sequence)
            # Creazione dei kmers a partire dalle sequenze
            for slide in range(0, len(sequence)-kmer_size, slide_step):
                sequence_words.add(sequence[slide:slide+kmer_size])


def one_hot_encoding_kmers():
    '''OneHotEncoder applicato sui kmers. '''
    print("Inizio fase OneHotEncoding dei dati")
    # Dizionario per mapping da tripla a int
    sequence_words_list = list(sequence_words)
    sequence_words_list.sort()
    print("\t*Conversione triplette in OneHotEncoding")
    # integer encode
    integer_encoded = label_encoder.fit_transform(sequence_words_list)
    # binary encode
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    for i in range(0, len(sequence_words_list)):
        word_ohe[sequence_words_list[i]] = onehot_encoded[i]


def conversione_sequenze_2D():
    ''' Conversione delle sequenze di DNA in matrici 2D. La sequenza viene suddivisa in kmers, dunque si incolonnano le codifiche OHE delle coppie di kmers contigue. '''
    print("\t*Conversione delle sequenze in matrici 2D")
    final_sequences = {}
    index = 1
    for seq in sequences:
        final_seq = None
        # Costruisco la matrice 2D iterando sui kmer della sequenza seq
        for slide in range(0, len(seq)-kmer_size-1, 2):
            first_word = seq[slide:slide+kmer_size]         # Primo kmer
            second_word = seq[slide+1:slide+kmer_size+1]    # Secondo kmer
            first_ohe = word_ohe[first_word]
            second_ohe = word_ohe[second_word]
            # Colonna composta dall'OHE dei due kmer
            final_column = np.concatenate((first_ohe, second_ohe)).transpose()
            # Concatenazione delle colonne per formare la matrice 2D della sequenza
            if final_seq is None:
                final_seq = final_column
            else:
                final_seq = np.column_stack((final_seq, final_column))

        final_sequences[seq] = final_seq
        print(("\t\t%d/%d") % (index, len(sequences)), end="\r")
        index = index+1
    return final_sequences


def creazione_X():
    print("Creazione input X")
    final_sequences = conversione_sequenze_2D()
    x = list()
    index = 0
    for seq in sequences:
        x.append(final_sequences[seq])
    return np.array(x)


def creazione_Y():
    ''' Restituisci la coppia (y,types_np) in cui il primo rappresenta la versione categorical dei tipi, mentre il secondo l'output in formato intero. '''
    print("Conversione degli output Y in versione categorical")
    y = to_categorical(label_encoder.fit_transform(types))
    types_np = np.array(types)
    labels = np.unique(types_np)
    i = 0
    for l in labels:
        filter = types_np == l
        types_np[filter] = i
        i = i+1
    types_np = types_np.astype(int)
    return y, types_np


def carica_modello():
    response = (input("Vuoi caricare un modello allenato? y/n\n"))
    if response == "y":
        return True
    else:
        return False


def creazione_Modello(x, y):
    ''' Definizione modello convoluzione. Il modello presenta due livelli di convoluzione 2D,
    ciascuno seguito da un sub-sampling di tipo AveragePooling, e da una rete feedforward densa
    con 100 neuroni, al quale viene applicato un layer dropout per ridurre l'overfitting e il
    softmax per estrarre il risultato. '''
    height, width = x[0].shape
    _, num_classes = y.shape
    channels = 1
    f_size1 = 32  # numero filtri primo strato del modello
    f_size2 = 16  # numero filtri secondo strato del modello
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path+"weights", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True, mode="auto", save_freq="epoch"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10,
                                            verbose=1, mode="auto", baseline=None, restore_best_weights=False),
        # tf.keras.callbacks.TensorBoard(log_dir="./logs",histogram_freq=1,write_graph=True,write_images=True,update_freq="epoch",profile_batch=2,embeddings_freq=0,embeddings_metadata=None)
    ]
    model = Sequential()
    model.add(Conv2D(f_size1, kernel_size=3, activation="relu",
                        input_shape=(width, height, channels)))
    model.add(AveragePooling2D(
        pool_size=(2, 2), strides=None, padding='valid'
    ))
    model.add(Conv2D(f_size2, kernel_size=3, activation="relu"))
    model.add(AveragePooling2D(
        pool_size=(2, 2), strides=None, padding='valid'
    ))
    model.add(Flatten())
    model.add(Dense(100))   # hidden
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))  # output
    model.add(Softmax())
    model.compile(optimizer="adam",
                    loss="categorical_crossentropy", metrics=["accuracy"])
    return model, callbacks


def applica_Modello(train_index, test_index, load_model):
    ''' Esegue la fase di fitting e prediction su test e training sets. Se l'utente ha richiesto il caricamento di un modello già allenato,
    la fase di fitting viene saltata. '''
    # eseguiamo il reshape del dataset di training e di test
    height, width = x[0].shape
    channels = 1
    x_reshape = x.reshape(len(x), width, height, channels)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_train_reshaped = x_train.reshape(len(x_train), width, height, channels)
    x_test_reshaped = x_test.reshape(len(x_test), width, height, channels)
    # Verifico che non sia stato caricato un modello, per evitare di fare fitting
    if not load_model:
        history = model.fit(x_train_reshaped, y_train, callbacks=callbacks, epochs=epochs,
                            batch_size=batch_size, shuffle=True, validation_data=(x_test_reshaped, y_test))
    loss, acc = model.evaluate(x_test_reshaped, y_test)
    y_pred = model.predict(x_reshape)
    y_pred_class = np.argmax(y_pred, axis=1)
    # model.summary()
    confusion = confusion_matrix(
        labels=types_np, predictions=y_pred_class, num_classes=3)
    return y_pred, loss, acc, confusion


def calcola_Media(model, x, y):
    ''' Calcola la media dei risultati su un 10-fold. Stampa la media,
    il max e il min di accuracy/loss e le confusion matrix. '''
    n_splits = 10
    kfold = StratifiedKFold(n_splits)
    avg_loss = 0
    avg_acc = 0
    min_loss = None
    min_acc = None
    max_loss = None
    max_acc = None
    confusions = list()
    predictions = list()
    iterazione = 1
    load_model = carica_modello()
    if load_model:
        # model = tf.keras.models.load_model(model_path)
        print("Pesi del modello caricati!")
        model.load_weights(model_path+"weights")
    for train_index, test_index in kfold.split(x, types_np):
        y_pred, loss, acc, confusion = applica_Modello(train_index, test_index, load_model)
        logging.info((" === Iterazione: %d/%d ===\nLoss: %f\nAccuracy: %f\n[Matrice Confusione]") % (
            iterazione, n_splits, loss, acc))
        confusions.append(confusion)
        predictions.append(y_pred)
        if min_loss == None or min_loss > loss:
            min_loss = loss
        if max_loss == None or max_loss < loss:
            max_loss = loss
        if min_acc == None or min_acc > acc:
            min_acc = acc
        if max_acc == None or max_acc < acc:
            max_acc = acc
        avg_acc = avg_acc + acc
        avg_loss = avg_loss + loss
        iterazione = iterazione+1
        logging.info(confusion.numpy())
    logging.info(("[Accuracy]\nAverage: %f\nMax: %f\nMin: %f\n[Loss]\nAverage: %f\nMax: %f\nMin: %f") % (
        avg_acc/n_splits, max_acc, min_acc, avg_loss/n_splits, max_loss, min_loss))


lettura_file("splice.data")
one_hot_encoding_kmers()
x = creazione_X()
y, types_np = creazione_Y()
model, callbacks = creazione_Modello(x, y)
calcola_Media(model, x, y)
