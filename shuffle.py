import tensorflow as tf
from tqdm import tqdm

index = open("fimfiction.train.index").read().splitlines()

dataset = tf.data.Dataset.from_tensor_slices(index)
dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=128, num_parallel_calls=tf.data.experimental.AUTOTUNE)

d = dataset.shuffle(10000).prefetch(100)

i = 0
for idx, example in enumerate(tqdm(d)):
    if idx % 100000 == 0:
        try:
            writer.close()
        except:
            pass
        writer = tf.io.TFRecordWriter(f"F:/TF_RECORDS_SHUFFLED/fimfiction_{i}.tfrecords")
        i += 1
    writer.write(example.numpy())

writer.close()