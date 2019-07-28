from keras.preprocessing.image import ImageDataGenerator


class PrepareData:

    def __init__(self):
        self.data_generator_train = ImageDataGenerator()
        self.data_generator_test = ImageDataGenerator()

        # number of images to feed in the CNN
        self.batch_size = 32

        # image size 48*48 pixels
        self.img_size = 48

        # input path for the images
        self.path = "dataset/"

    def run(self):
        train_generator = self.data_generator_train.flow_from_directory(self.path + "train",
                                                                        target_size=(self.img_size, self.img_size),
                                                                        color_mode="grayscale",
                                                                        batch_size=self.batch_size,
                                                                        class_mode='categorical',
                                                                        shuffle=True)

        test_generator = self.data_generator_test.flow_from_directory(self.path + "test",
                                                                      target_size=(self.img_size, self.img_size),
                                                                      color_mode="grayscale",
                                                                      batch_size=self.batch_size,
                                                                      class_mode='categorical',
                                                                      shuffle=False)

        return train_generator, test_generator
