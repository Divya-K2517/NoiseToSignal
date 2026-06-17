#testing the neural network on the Pulsar Dataset HTRU2
#https://www.kaggle.com/datasets/charitarth/pulsar-dataset-htru2

import os
import urllib.request
import numpy as np
from p1 import compute_accuracy, Layer_Dense, Layer_Dropout,Activation_ReLU, Activation_LeakyReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD, clip_gradients, Activation_Softmax, Loss_CategoricalCrossentropy, Optimizer_Adam

#download dataset
DATA_DIR = "nsl_kdd_data"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
}

for filename, url in FILES.items():
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, local_path)
    else:
        print(f"{filename} already downloaded, skipping.")
path = DATA_DIR
print("Path to dataset files:", path)
print("Files found:", os.listdir(path))

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack_type", "difficulty"
]
CATEGORICAL_COLS = ["protocol_type", "service", "flag"] 

#there are 22 types of attacks, but we will collapse them into 5 categories for ease
ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "worm": "dos", "mailbomb": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L (remote to local)
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    # U2R (user to root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"] #5 possible classes
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

#get the rows from the downloaded NSL-KDD txt files
def load_nslkdd_split(filepath):
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().strip(".") #some files end lines with a trailing "."
            if not line:
                continue
            rows.append(line.split(","))
    return rows
train_rows = load_nslkdd_split(os.path.join(DATA_DIR, "KDDTrain+.txt"))
test_rows = load_nslkdd_split(os.path.join(DATA_DIR, "KDDTest+.txt"))

#first 41 columns are features, column 41 is the attack label
train_features_raw = [row[:41] for row in train_rows]
train_labels_raw = [row[41] for row in train_rows]
test_features_raw = [row[:41] for row in test_rows]
test_labels_raw = [row[41] for row in test_rows]

#map the 22 labels to one of 5 categories
def map_labels(raw_labels):
    mapped = []
    for label in raw_labels:
        category = ATTACK_MAP.get(label, "r2l") #unseen attack names default to r2l bucket
        mapped.append(CLASS_TO_IDX[category])
    return np.array(mapped, dtype=np.int64)

y_train = map_labels(train_labels_raw)
y_test = map_labels(test_labels_raw)

#figure out which of the 41 feature columns are categorical (text) vs numeric
feature_names = COLUMN_NAMES[:41]
cat_idx = [feature_names.index(c) for c in CATEGORICAL_COLS]
num_idx = [i for i in range(41) if i not in cat_idx]
 
#build a vocabulary for each categorical column using only the train set
vocabs = {}
for ci in cat_idx:
    values = sorted(set(row[ci] for row in train_features_raw))
    vocabs[ci] = {v: i for i, v in enumerate(values)}
 
def one_hot_encode_column(features_raw, col_idx):
    vocab = vocabs[col_idx]
    n = len(features_raw)
    out = np.zeros((n, len(vocab)), dtype=np.float32)
    for row_i, row in enumerate(features_raw):
        val = row[col_idx]
        if val in vocab:
            out[row_i, vocab[val]] = 1.0
        #unseen category at test time -> leave as all-zero row
    return out

def build_feature_matrix(features_raw):
    numeric = np.array(
        [[float(row[i]) for i in num_idx] for row in features_raw],
        dtype=np.float32
    )
    cat_blocks = [one_hot_encode_column(features_raw, ci) for ci in cat_idx]
    return numeric, np.concatenate(cat_blocks, axis=1)
 
x_train_num, x_train_cat = build_feature_matrix(train_features_raw)
x_test_num, x_test_cat = build_feature_matrix(test_features_raw)

#min-max scale numeric columns using train min/max (this is our version of /255.0 normalization)
col_min = x_train_num.min(axis=0, keepdims=True)
col_max = x_train_num.max(axis=0, keepdims=True)
col_range = np.maximum(col_max - col_min, 1e-8) #avoid divide-by-zero for constant columns
 
x_train_num = (x_train_num - col_min) / col_range
x_test_num = (x_test_num - col_min) / col_range
x_test_num = np.clip(x_test_num, 0.0, 1.0) #test values might fall slightly outside train's range
 
#combine numeric + one-hot categorical into the final feature vectors
images = np.concatenate([x_train_num, x_train_cat], axis=1).astype(np.float32) #train features, kept var name for minimal diff
x_train = images
x_test = np.concatenate([x_test_num, x_test_cat], axis=1).astype(np.float32)

#split into test and train
np.random.seed(0)
indices = np.random.permutation(len(x_train)) #getting random indices to shuffle the data
split = int(0.8 * len(images)) #80% for training, 20% for testing
x_train, y_train = x_train[indices], y_train[indices]

# #creating a subset for easier testing
# subset_size = 2000
# x_train, y_train = x_train[:subset_size], y_train[:subset_size]

numInputs = x_train.shape[1] #number of features (pixels)
numClasses = len(CLASS_NAMES) #number of classes

#define the layers of the neural network
dense1 = Layer_Dense(numInputs, 128) #128 is the length of the output vector from this layer, which is also the number of neurons in this layer
activation1 = Activation_LeakyReLU()
dropout1 = Layer_Dropout(0.5)

dense2 = Layer_Dense(128, 64) #128 is the length of the output vector from this layer, which is also the number of neurons in this layer
activation2 = Activation_LeakyReLU()
dropout2 = Layer_Dropout(0.5)

dense3 = Layer_Dense(64, numClasses) #output layer here is 10 (numClasses)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.0001)

epochs = 20
batch_size = 64

if __name__ == "__main__":
    for epoch in range(epochs):
        #each epoch we train a single batch

        #shulffe the training data
        indices = np.random.permutation(len(x_train))
        x_shuffled, y_shuffled = x_train[indices], y_train[indices]

        epochLoss = 0
        epochAccuracy = 0
        numBatches = len(x_train) // batch_size

        for i in range(numBatches):
            #to get the batch, we do the 1, 2, 3, ... batch_size, then the next batch_size, etc
            x_batch = x_shuffled[i*batch_size:(i+1)*batch_size]
            y_batch = y_shuffled[i*batch_size:(i+1)*batch_size]

            #forward pass
            dense1.forward(x_batch)
            activation1.forward(dense1.output)
            dropout1.forward(activation1.output, training=True)
            dense2.forward(dropout1.output)
            activation2.forward(dense2.output)
            dropout2.forward(activation2.output, training=True)
            dense3.forward(dropout2.output)

            loss = loss_activation.forward(dense3.output, y_batch)
            epochLoss += loss

            predictions = np.argmax(loss_activation.output, axis=1)
            epochAccuracy += np.mean(predictions == y_batch)
            
            #backward pass
            loss_activation.backward(loss_activation.output, y_batch)
            dense3.backward(loss_activation.dinputs)
            dropout2.backward(dense3.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            dropout1.backward(dense2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)
            
            #clip gradients
            clip_gradients(dense1)
            clip_gradients(dense2)
            clip_gradients(dense3)

            #update weights and biases
            optimizer.update_params(dense1)
            optimizer.update_params(dense2)
            optimizer.update_params(dense3)
        
        dead_ratio = np.mean(activation1.output == 0)
        print(f"Fraction of dead ReLU outputs in layer 1: {dead_ratio:.2%}")
        print(f"Epoch {epoch+1:>2}/{epochs}  loss: {epochLoss/numBatches:.4f}  acc: {epochAccuracy/numBatches*100:.1f}%")
        print()

    #testing the model on the test set
    dense1.forward(x_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)

    activation3 = Activation_Softmax()
    activation3.forward(dense3.output)

    loss_fn = Loss_CategoricalCrossentropy()
    test_loss = loss_fn.calculate(activation3.output, y_test)
    test_acc = compute_accuracy(activation3.output, y_test)

    print(f"\n── Test set evaluation ──")
    print(f"Loss:     {test_loss:.4f}")
    print(f"Accuracy: {test_acc*100:.1f}%")

    # after test evaluation
    for i, name in enumerate(CLASS_NAMES):
        mask = (y_test == i)
        if mask.sum() == 0:
            continue
        class_preds = np.argmax(activation3.output[mask], axis=1)
        class_acc = np.mean(class_preds == i)
        print(f"  [{i}] {name:<35} acc: {class_acc*100:.1f}%  (n={mask.sum()})")

    conf_matrix = np.zeros((numClasses, numClasses), dtype=int)
    predicted = np.argmax(activation3.output, axis=1)
    for true, pred in zip(y_test, predicted):
        conf_matrix[true][pred] += 1

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = "     " + "".join(f"{i:>6}" for i in range(numClasses))
    print(header)
    for i, row in enumerate(conf_matrix):
        print(f"[{i}]  " + "".join(f"{v:>6}" for v in row))