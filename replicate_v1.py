;; Happy hacking, Nasy - Emacs ♥ you!

from csv import reader, writer
from itertools import chain
from tqdm import tqdm
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras

def read_row_dataset(path = "./waveforms_with_paraName.csv"):
    with open(path, newline="") as f:
        next(f)
        for row in reader(f):
            yield row[1:18607], row[18607:]

with open(f"wave_x", "w") as xf, open(f"wave_y", "w") as yf:
    xw = writer(xf)
    yw = writer(yf)
    for x, y in tqdm(read_row_dataset()):
        xw.writerow(x)
        yw.writerow(y)

def read_dataset(prefix = "wave"):
    return train_test_split(genfromtxt(f"{prefix}_x", delimiter = ","), genfromtxt(f"{prefix}_y", delimiter = ","))

train_x, test_x, train_y, test_y = read_dataset()

stdxs = StandardScaler().fit(train_x)
stdys = StandardScaler().fit(train_y)

train_x = np.expand_dims(stdxs.transform(train_x), axis=1)
train_y = stdys.transform(train_y)

test_x = np.expand_dims(stdxs.transform(test_x), axis=1)
test_y = stdys.transform(test_y)


print(train_x.shape)

model = keras.Sequential(
    [
        #keras.Sequential([
            keras.layers.Conv1D(filters = 64, kernel_size = 16, strides = 1, padding = "causal", input_shape = (1, 18606)),
            keras.layers.MaxPool1D(pool_size = 4, strides = 4, padding = "same"),
            keras.layers.ReLU(),
#        ]),
#        keras.Sequential([
            keras.layers.Conv1D(filters = 128, kernel_size = 16, strides = 1, padding = "causal"),
            keras.layers.MaxPooling1D(pool_size = 4, strides = 4, padding = "same"),
            keras.layers.ReLU(),
#        ]),
#        keras.Sequential([
            keras.layers.Conv1D(filters = 256, kernel_size = 16, strides = 1, padding = "causal"),
            keras.layers.MaxPooling1D(pool_size = 4, strides = 4, padding = "same"),
            keras.layers.ReLU(),
#        ]),
#        keras.Sequential([
            keras.layers.Conv1D(filters = 512, kernel_size = 32, strides = 1, padding = "causal"),
            keras.layers.MaxPooling1D(pool_size = 4, strides = 4, padding = "same"),
            keras.layers.ReLU(),
#        ]),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation = "relu"),
        keras.layers.Dense(64, activation = "relu"),
        keras.layers.Dense(9, activation = "relu"),
    ]
)
model.compile(optimizer='adam', loss='mse')
print(model.summary())

model.fit(train_x, train_y)

p = model.predict(test_x)

result = stdys.inverse_transform(p)

print(result, stdys.inverse_transform(test_y))
