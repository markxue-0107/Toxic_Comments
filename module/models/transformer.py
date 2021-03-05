import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = d_model
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension should be divisible by number of heads"

        self.depth = self.embed_dim // self.num_heads
        self.query_dense = layers.Dense(self.embed_dim)
        self.value_dense = layers.Dense(self.embed_dim)
        self.key_dense = layers.Dense(self.embed_dim)
        self.dense_layer = layers.Dense(self.embed_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # scale
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention, weights = self.scaled_dot_product_attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        output = self.dense_layer(concat_attention)
        return output


class EncoderLayer(layers.Layer):
    def __init__(self, embed_dim, nums_head, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, nums_head)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(embed_dim)])

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attention_output = self.mha(inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)

        return self.layernorm2(out1 + ffn_out)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim, pretrained_embedding=None):
        super(TokenAndPositionEmbedding, self).__init__()
        if pretrained_embedding is not None:
            self.token_embed = layers.Embedding(input_dim=vocab_size,
                                                output_dim=embed_dim,
                                                weights=[pretrained_embedding],
                                                input_length=max_len,
                                                trainable=False)
        else:
            self.token_embed = layers.Embedding(input_dim=vocab_size,
                                                output_dim=embed_dim,
                                                input_length=max_len,
                                                embeddings_initializer='uniform',
                                                trainable=True)

        self.pos_embed = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions


class Encoder(layers.Layer):
    def __init__(self, max_len, nums_layer, embed_dim, nums_head, dff, vocab_size,
                 pretrained_embedding, rate=0.1):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.nums_layer = nums_layer
        self.embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim, pretrained_embedding)
        self.enc_layers = [EncoderLayer(embed_dim, nums_head, dff, rate) for _ in range(nums_layer)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training):
        x = self.embedding_layer(x)
        for i in range(self.nums_layer):
            x = self.enc_layers[i](x, training)

        return x


class Transformer(object):
    def __init__(self, config, classes, vocab_size, logger, embedding_matrix):
        self.models = {}
        self.logger = logger
        self.vocab_size = vocab_size
        self.classes = classes
        self.config = config
        self.pretrained_embedding = embedding_matrix
        self.nums_classes = len(classes)
        self.model = self._build()

    def _show_training_config_para(self):
        title = "Training config parameters"
        self.logger.info(title.center(40, '-'))
        self.logger.info("---model_name = {}".format(self.config['model_name']))
        self.logger.info("---max_input_len = {}".format(self.config['max_len']))
        self.logger.info("---batch_size = {}".format(self.config['batch_size']))
        self.logger.info("---dropout = {}".format(self.config['dropout']))
        self.logger.info("---epochs = {}".format(self.config['epochs']))
        self.logger.info("---num_of_classes = {}".format(self.nums_classes))
        self.logger.info("---dff = {}".format(self.config['dff']))
        self.logger.info("---nums_head = {}".format(self.config['nums_head']))
        self.logger.info("---nums_layer = {}".format(self.config['nums_layer']))

    def _build(self):
        self._show_training_config_para()
        embedding_dim = self.config['embedding_col']
        max_len = self.config['max_len']
        dff = self.config['dff']
        nums_head = self.config['nums_head']
        vocab_size = self.vocab_size
        drop_out_rate = self.config['dropout']
        nums_layer = self.config['nums_layer']

        inputs = layers.Input(shape=(max_len,))
        encoder = Encoder(max_len, nums_layer, embedding_dim, nums_head, dff,
                          vocab_size, self.pretrained_embedding)
        x = encoder(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Dense(self.nums_classes, activation=None)(x)
        outputs = layers.Dense(self.nums_classes, activation='sigmoid')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):
        history = self.model.fit(train_x,
                                 train_y,
                                 epochs=self.config['epochs'],
                                 verbose=True,
                                 validation_data=(validate_x, validate_y),
                                 batch_size=self.config['batch_size'])

        predictions = self.predict(validate_x)
        return predictions, history

    def predict_prob(self, test_x):
        return self.model.predict(test_x)

    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return probs >= 0.5