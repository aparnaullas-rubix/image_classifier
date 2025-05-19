import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf



# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.utils import to_categorical
#
# (x_train, y_train), (x_val, y_val) = cifar10.load_data()
# x_train = x_train / 255
# x_val = x_val / 255
#
# y_train = to_categorical(y_train, 10)
# y_val = to_categorical(y_val, 10)
#
# model = Sequential([
#     Flatten(input_shape=(32, 32, 3)),
#     Dense(1000, activation='relu'),
#     Dense(10, activation='softmax')
# ])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val))
# model.save('cifar_model.h5')

def main():
    st.title("CIFAR10 Web Classifier")
    st.write("Upload any image for the classifier")
    file = st.file_uploader("Please upload an image (jpeg, png)", type=["jpg", "png"])
    if file:
        image = Image.open(file)
        st.image(image,  use_column_width=True)

        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape(1, 32, 32, 3)

        model = tf.keras.models.load_model('cifar_model.h5')
        prediction = model.predict(img_array)
        cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        figure, axis = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        axis.barh(y_pos, prediction[0], align="center")
        axis.set_yticks(y_pos)
        axis.set_yticklabels(cifar10_classes)
        axis.invert_yaxis()
        axis.set_xlabel("Prediction")
        axis.set_title("CIFAR10 Prediction")
        st.pyplot(figure)
    else:
        st.text("Please upload a proper image")

if __name__ == '__main__':
    main()