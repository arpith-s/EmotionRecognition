from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import os


class CNN:
    def __init__(self, train_generator, test_generator):
        self.epochs = 50
        self.classes = 7
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.model = Sequential()

    def create(self):
        # 1st Convolution
        self.model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # 2nd Convolution layer
        self.model.add(Conv2D(128, (5, 5), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # 3rd Convolution layer
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # 4th Convolution layer
        self.model.add(Conv2D(512, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Flattening
        self.model.add(Flatten())

        # 1st Fully connected layer
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        # 2nd Fully connected layer
        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(self.classes, activation='softmax'))

        opt = Adam(lr=0.0001)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit_generator(generator=self.train_generator,
                                           steps_per_epoch=self.train_generator.n // self.train_generator.batch_size,
                                           epochs=self.epochs,
                                           validation_data=self.test_generator,
                                           validation_steps=self.test_generator.n // self.test_generator.batch_size)

        if not os.path.exists('model'):
            os.makedirs('model')
        # Save Model
        self.model.save('model/CNN.h5')

        return history
