from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.optimizers import Adam
#load_data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#preprocess_data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#build the architecture
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
#compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
#history=model.fit(x_train, y_train, epochs=10, batch_size=64)
#history=model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
#history=model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test))
history=model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test))
#print
print(history.history.items())
print(history.history.keys())
#evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f'accuracy.{accuracy},loss.{loss}')
#visualize
plt.plot(history.history['accuracy'], label='train accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='val accuracy', color='red')
plt.legend()
plt.title('epoch vs accuracy on train and test data')
plt.show()
plt.plot(history.history['loss'], label='train loss', color='blue')
plt.plot(history.history['val_loss'], label='val loss', color='red')
plt.legend()
plt.title('epoch vs loss on train and test data')
plt.show()