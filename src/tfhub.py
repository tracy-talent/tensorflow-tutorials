import tensorflow as tf
import tensorflow_hub as hub

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
tokens_input = [["the", "cat", "is", "on", "the", "mat"],
                ["dogs", "are", "in", "the", "fog", ""]]
tokens_length = [6, 5]
embeddings = elmo(
    inputs={
        "tokens": tokens_input,
        "sequence_len": tokens_length
    },
    signature="tokens",
    as_dict=True)["elmo"]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(embeddings.eval())
