import tensorflow as tf 

rep_ds = tf.data.Dataset.list_files("train_samples/content/train_samples/*.jpg")
HEIGHT, WIDTH = 320, 320

def representative_dataset_gen():
    for image_path in rep_ds:
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        resized_img = tf.image.resize(img, (HEIGHT, WIDTH))
        resized_img = resized_img[tf.newaxis, :]
        yield [resized_img]