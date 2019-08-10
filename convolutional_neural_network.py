from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import crayons as cr
import os


class CNN:
    def __init__(self, train_generator, test_generator):
        self.epochs = 5
        self.classes = 7
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.model = Sequential()

    def create(self):
        try:
            # 1st Convolution

            self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(128, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(128, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Flatten())
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dense(self.classes, activation='sigmoid'))

            # opt = RMSprop(lr=1e-4)
            opt = Adam()
            self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

            # Set a learning rate annealer
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                        patience=3,
                                                        verbose=1,
                                                        factor=0.5,
                                                        min_lr=0.00001)

            history = self.model.fit_generator(generator=self.train_generator,
                                               steps_per_epoch=self.train_generator.n // self.train_generator.batch_size,
                                               epochs=self.epochs,
                                               validation_data=self.test_generator,
                                               validation_steps=self.test_generator.n // self.test_generator.batch_size,
                                               callbacks=[learning_rate_reduction])

            if not os.path.exists('models'):
                os.makedirs('models')
            # Save Model
            self.model.save('models/CNN.h5')

            return history
        except Exception as e:
            print()
            print(cr.red(str(e), bold=True))
            exit(1)
