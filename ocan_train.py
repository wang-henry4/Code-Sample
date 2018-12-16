import tensorflow as tf
import numpy as np
from itertools import count
from config import *
from utils import *
from pathlib import Path
from ocan_base import Ocan_Base

class OCAN(Ocan_Base):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if RECALCULATE_MIN_MAX:
            save_bounds(TRAIN_PATH)

        with self.g.as_default(), self.sess.as_default():
            X_MAX, X_MIN = get_bounds()
            # Get Dataset for pre-train net(for density estimation)
            features_tar, label_tar = get_dataset(TRAIN_PATH,
                                                batch_size=GAN_BATCH_SIZE).make_one_shot_iterator().get_next()

            # Get Dataset for ocan(benigns/real data; only one class)
            train_dataset = get_dataset(TRAIN_PATH, GAN_EPOCHS, GAN_BATCH_SIZE)
            # test_dataset = get_dataset(TEST_PATH, epochs=1, batch_size=10000)

            self.features, label = train_dataset.make_one_shot_iterator().get_next()

            # Process data for pre-train net T and discriminator D
            if CATEGORICAL_FEATURES:
                X_cat_tar = one_hot_encode(features_tar)
                X_cat  = one_hot_encode(self.features)

            X_cont_tar = get_continuous_features(features_tar)
            label_tar = tf.one_hot(tf.cast(label_tar, tf.int32), 2, axis=1)

            X_cont = get_continuous_features(self.features)
            label = tf.one_hot(tf.cast(label, tf.int32), 2, axis=1)

            self.encoder = tf.keras.models.load_model(ENCODER_PATH)

            for layer in self.encoder.layers:
                layer.trainable = False        

            # Placeholders
            self.Z = tf.placeholder(tf.float32, shape=[None, noise_dim],name='Z')
            self.y_gen = tf.placeholder(tf.int32, shape=[None, DISCRIMINATOR_DIMS[-1]], name='y_gen')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            # scaler
            x_min = tf.Variable(initial_value=X_MIN, trainable=False, name='x_min')
            x_max = tf.Variable(initial_value=X_MAX, trainable=False, name='x_max')
            theta_min_max = [x_min, x_max]

            # Build neural networks
            def apply_encoder(x_cat, x_cont):
                x_cont = tf.cast(x_cont, tf.float32)
                enc2 = self.encoder(x_cat, training=False)
                encoding = tf.concat([enc2, x_cont], axis=1)
                enc_output = minmax_scaler(encoding, x_min, x_max)
                return enc_output

            # (Real data) -> D -> probility
            if CATEGORICAL_FEATURES:
                self.D_prob_real, D_logit_real, D_h2_real = self.discriminator(apply_encoder(X_cat, X_cont), 
                                                                               DISCRIMINATOR_DIMS,
                                                                               'discriminator')
                D_prob_tar, D_logit_tar, D_h2_tar = self.discriminator(apply_encoder(X_cat_tar, X_cont_tar), 
                                                                       DISCRIMINATOR_DIMS, 
                                                                       'discriminator_t')
            else:
                self.D_prob_real, D_logit_real, D_h2_real = self.discriminator(minmax_scaler(tf.cast(X_cont, tf.float32),
                                                                                x_min, x_max), 
                                                                               DISCRIMINATOR_DIMS, 
                                                                               'discriminator')

                D_prob_tar, D_logit_tar, D_h2_tar = self.discriminator(minmax_scaler(tf.cast(X_cont_tar, tf.float32),
                                                                                    x_min, x_max), 
                                                                       DISCRIMINATOR_DIMS, 
                                                                       'discriminator_t')

            # (Noise Z) -> G -> (G_sample) -> D -> Probility
            GC_sample = self.generator(self.Z, GENERATOR_DIMS, 'generator_c')
            G_sample = self.generator(self.Z, GENERATOR_DIMS, 'generator')
            D_prob_gen, D_logit_gen, D_h2_gen = self.discriminator(GC_sample, DISCRIMINATOR_DIMS, 'discriminator')

            # (Real data) -> T -> Probility(density estimation)
            D_prob_tar_genC, D_logit_tar_genC, D_h2_tar_genC = self.discriminator_tar(GC_sample, DISCRIMINATOR_DIMS, 'discriminator_t')

            # Loss functions
            # loss of discriminator_D
            D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_real, labels=label))
            D_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_gen, labels=self.y_gen))
            # ent_real_loss: a conditional entropy term
            # which encourages the discriminator to detect real benign users with high confidence
            self.ent_real_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(self.D_prob_real, tf.log(self.D_prob_real)), 1))
            # ent_gen_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(D_prob_gen, tf.log(D_prob_gen)), 1))
            self.D_loss = D_loss_real + D_loss_gen + 1.9*self.ent_real_loss

            # loss of discriminator_T
            self.T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_tar, labels=label_tar))

            # loss of generator
            #  pt term increases the diversity of G_sample and can be considered as a proxy for minimizing −H(pG)
            pt_loss = pull_away_loss(D_h2_tar_genC)
            G_ent_loss = compliment_entropy_loss(D_prob_tar_genC)
            # feature matching loss: to ensure G_sample are constrained in the space of user representation
            self.fm_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(D_logit_real - D_logit_gen), 1)))
            self.G_loss = pt_loss + G_ent_loss + self.fm_loss

            # Optimizer
            self.theta_D = self.get_variables('discriminator')
            self.theta_T = self.get_variables('discriminator_t')
            self.theta_G = self.get_variables('generator')
            self.theta_GC = self.get_variables('generator_c')

            self.D_optimizer = tf.train.AdamOptimizer()
            self.GC_optimizer = tf.train.AdamOptimizer()
            self.T_optimizer = tf.train.AdamOptimizer()

            self.D_solver = self.D_optimizer.minimize(self.D_loss, var_list=self.theta_D)
            self.GC_solver = self.GC_optimizer.minimize(self.G_loss, var_list=self.theta_G)
            self.T_solver = self.T_optimizer.minimize(self.T_loss, var_list=self.theta_T)

            all_vars = self.theta_D+self.theta_G+self.theta_T+theta_min_max+\
                        self.D_optimizer.variables()+self.GC_optimizer.variables()+self.T_optimizer.variables()
            # Global variables initializer
            self.init_op = tf.variables_initializer(all_vars)

    def train(self):
        with self.g.as_default(), self.sess.as_default():
            self.sess.run(self.init_op)

            print('checking encoder weights')
            # Labels of fake data(G_sample)
            y_fake_mb = one_hot(np.ones(GAN_BATCH_SIZE), 2)

            # Pre-train net
            while True:
                try:
                    _ = self.sess.run(self.T_solver, feed_dict={self.keep_prob: 1.0})
                except tf.errors.OutOfRangeError:
                    print('T Discriminator Training Complete')
                    break

            # Train gan
            print('Training')
            for i in count():
                try:
                    # Fix G，train D
                    for _ in range(DISCRIMINATOR_K):
                        _, D_loss_curr, ent_real_curr = self.sess.run([self.D_solver, self.D_loss, self.ent_real_loss],
                                                                    feed_dict={
                                                                        self.Z: sample_Z(GAN_BATCH_SIZE, noise_dim),
                                                                        self.y_gen: y_fake_mb,
                                                                        self.keep_prob: 1.0
                                                                    })
                    # Fix D，train G
                    _, G_loss_curr, fm_loss_curr = self.sess.run([self.GC_solver, self.G_loss, self.fm_loss],
                                                            feed_dict={
                                                                self.Z: sample_Z(GAN_BATCH_SIZE, noise_dim),
                                                                self.keep_prob: 1.0
                                                            })
                    if not i % PRINT_FREQ:
                        print('batch: ', i, ', loss of discriminator: ', D_loss_curr)

                except tf.errors.OutOfRangeError:
                    print('Training finished')
                    break

    def train_save(self):
        self.train()
        self.save(GAN_WEIGHTS_DIR, self.theta_D)

class LSTM_AE_OCAN(Ocan_Base):
    def __init__(self, pre_encoding, *args, **kwargs):
        super().__init__()
        # controls whether to use trained encoder or pre encoded files
        self.with_encoder = not pre_encoding

        if not self.with_encoder:
            # load pre encoded data
            self.encoded_data = np.load(LSTM_ENCODED_DATA)

        with self.g.as_default(), self.sess.as_default():
            # Placeholders
            if self.with_encoder:
                feature_shape = [None, *LSTM_INPUT_DIM]
            else:
                feature_shape = [None, LSTM_ENCODING_DIM[-1]]
            label_shape = [None, DISCRIMINATOR_DIMS[-1]]

            self.features_tar = tf.placeholder(tf.float32, shape=feature_shape, name='features_tar')
            self.label_tar = tf.placeholder(tf.int32, shape=label_shape, name='label_tar')

            self.features = tf.placeholder(tf.float32, shape=feature_shape, name='features')
            self.label = tf.placeholder(tf.int32, shape=label_shape, name='label')
            print(f'with encoder: {self.with_encoder}')
            if self.with_encoder:
                print('LSTM_PATH: ' + LSTM_ENCODER_PATH)
                self.encoder = tf.keras.models.load_model(LSTM_ENCODER_PATH)
                for layer in self.encoder.layers:
                    layer.trainable = False
                print('ENCODER OUTPUT: ' + str(self.encoder.output.shape))
            self.Z = tf.placeholder(tf.float32, shape=[None, noise_dim],name='Z')
            self.y_gen = tf.placeholder(tf.int32, shape=label_shape, name='y_gen')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            def apply_encoder(x_cat):
                if self.with_encoder:
                    # print(self.encoder.summary())
                    # print(x_cat.shape)
                    enc_output = self.encoder(x_cat, training=False)
                    # print(enc_output.shape)
                else:
                    enc_output = x_cat
                return enc_output

            ### BUILD NETWORKS ###
            # regular gan(generator，discriminator_t) 的作用是估计real data 的分布；
            # complementary gan（generator_c, discriminator）则是最后要训练的ocan

            G_sample = self.generator(self.Z, GENERATOR_DIMS, 'generator')
            GC_sample = self.generator(self.Z, GENERATOR_DIMS, 'generator_c')

            self.D_prob_real, D_logit_real, D_h2_real = self.discriminator(apply_encoder(self.features), 
                                                                           DISCRIMINATOR_DIMS,
                                                                           'discriminator')
            D_prob_tar, D_logit_tar, D_h2_tar = self.discriminator(apply_encoder(self.features_tar), 
                                                                   DISCRIMINATOR_DIMS,
                                                                   'discriminator_t')


            D_prob_tar_gen, D_logit_tar_gen, D_h2_tar_gen = self.discriminator(G_sample, 
                                                                               DISCRIMINATOR_DIMS,
                                                                               'discriminator_t')
            # (Noise Z) -> GC -> (GC_sample) -> D -> Probility
            D_prob_gen, D_logit_gen, D_h2_gen = self.discriminator(GC_sample, DISCRIMINATOR_DIMS, 'discriminator')

            # (Real data) -> T -> Probility(density estimation)
            D_prob_tar_genC, D_logit_tar_genC, D_h2_tar_genC = self.discriminator(GC_sample, DISCRIMINATOR_DIMS, 'discriminator_t')
            ### ----------------------- ###
            ### LOSS ###
            # loss of discriminator_D
            D_loss_real = tf.losses.softmax_cross_entropy(self.label, logits=D_logit_real)
            D_loss_gen = tf.losses.softmax_cross_entropy(self.y_gen, logits=D_logit_gen)

            # D_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_real, labels=self.label))
            # D_loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_gen, labels=self.y_gen))

            # ent_real_loss: a conditional entropy term
            # which encourages the discriminator to detect real benign users with high confidence
            self.ent_real_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(self.D_prob_real, tf.log(self.D_prob_real)), 1))
            # ent_gen_loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(D_prob_gen, tf.log(D_prob_gen)), 1))
            self.D_loss = D_loss_real + D_loss_gen + 1.9*self.ent_real_loss

            # loss of discriminator_T and generator_C (the regular gan loss)
            #self.T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logit_tar, labels=self.label_tar))
            self.T_loss = -tf.reduce_mean(tf.log(D_prob_tar) + tf.log(1.0 - D_prob_tar_gen))
            self.G_loss = -tf.reduce_mean(tf.log(D_prob_tar_gen))

            # loss of generator (the OCAN)
            #  pt term increases the diversity of G_sample and can be considered as a proxy for minimizing −H(pG)
            pt_loss = pull_away_loss(D_h2_tar_genC)
            G_ent_loss = compliment_entropy_loss(D_prob_tar_genC)
            # feature matching loss: to ensure G_sample are constrained in the space of user representation
            self.fm_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(D_logit_real - D_logit_gen), 1)))
            self.GC_loss = pt_loss + G_ent_loss + self.fm_loss
            ### --------------------------- ###
            ### OPTIMIZERS ###
            self.theta_D = self.get_variables('discriminator')
            self.theta_T = self.get_variables('discriminator_t')
            self.theta_G = self.get_variables('generator')
            self.theta_GC = self.get_variables('generator_c')

            self.D_optimizer = tf.train.AdamOptimizer()
            self.GC_optimizer = tf.train.AdamOptimizer()
            self.G_optimizer = tf.train.AdamOptimizer()
            self.T_optimizer = tf.train.AdamOptimizer()

            self.D_solver = self.D_optimizer.minimize(self.D_loss, var_list=self.theta_D)
            self.GC_solver = self.GC_optimizer.minimize(self.GC_loss, var_list=self.theta_GC)
            self.G_solver = self.G_optimizer.minimize(self.G_loss, var_list=self.theta_G)
            self.T_solver = self.T_optimizer.minimize(self.T_loss, var_list=self.theta_T)
            ### ---------------------------- ###
            all_vars = self.theta_D + self.theta_GC + self.theta_T + self.theta_G\
                       +self.D_optimizer.variables() + self.GC_optimizer.variables()\
                       +self.T_optimizer.variables() + self.G_optimizer.variables()

            # Global variables initializer
            self.init_op = tf.variables_initializer(all_vars)
            # Saver to store weights and biases of ocan
            #self.saver = tf.train.Saver(theta_D)

    def get_data(self):
        """generator object that yields batches of training data
        """
        if self.with_encoder:
            for i in count():
                batchdata = pd.read_csv(SEQUENTIAL_TRAIN_PATH,
                                                    nrows=GAN_BATCH_SIZE,
                                                    skiprows=i * GAN_BATCH_SIZE + 1,
                                                    names=SEQUENTIAL_COLUMN_NAMES.keys(),
                                                    dtype=SEQUENTIAL_COLUMN_NAMES)
                if len(batchdata) < GAN_BATCH_SIZE:
                    yield None
                batchdata = batchdata['seq_contents'].values
                yield get_data_for_lstm_ae(batchdata)
        else:
            # shuffles data
            self.encoded_data = self.encoded_data[np.random.permutation(self.encoded_data.shape[0])]
            for i in count():
                result = self.encoded_data[i*GAN_BATCH_SIZE:(i+1)*GAN_BATCH_SIZE,:]
                if result.shape[0] < GAN_BATCH_SIZE:
                    yield None
                yield result
    
    def train(self):
        with self.g.as_default(), self.sess.as_default():
            tf.set_random_seed(10)
            np.random.seed(0)
            self.sess.run(self.init_op)

            print('checking encoder weights')
            # labels: 1 for abnormals(vandals) and 0 for normals(benigns)
            y_fake_mb = one_hot(np.ones(GAN_BATCH_SIZE), 2)
            y_real_mb = one_hot(np.zeros(GAN_BATCH_SIZE), 2)

            # Pre-train net
            for _ in range(10):
                for inputdata in self.get_data():
                    if inputdata is None:
                        break
                    _ = self.sess.run(self.T_solver, feed_dict={self.features_tar: inputdata,
                                                                self.Z: sample_Z(GAN_BATCH_SIZE, noise_dim),
                                                                self.keep_prob: 1.0})
                    _ = self.sess.run(self.G_solver, feed_dict={self.Z: sample_Z(GAN_BATCH_SIZE, noise_dim),
                                                                self.keep_prob: 1.0})

            print('T Discriminator Training Complete')

            # Train gan
            print('Training')
            for _ in range(GAN_EPOCHS):
                print('{} epochs'.format(_))
                datagen = self.get_data()   # initialize data generator for training
                # one epoch train loop
                for i, inputdata in enumerate(datagen, 1):
                    if inputdata is None:
                        break

                    # Fix G，train D
                    # discriminator 多训练几个batch
                    if i % DISCRIMINATOR_K != 0:
                        _, D_loss_curr, ent_real_curr = self.sess.run([self.D_solver, self.D_loss, self.ent_real_loss],
                                                                    feed_dict={
                                                                        self.features: inputdata,
                                                                        self.label: y_real_mb,
                                                                        self.Z: sample_Z(GAN_BATCH_SIZE, noise_dim),
                                                                        self.y_gen: y_fake_mb,
                                                                        self.keep_prob: 1.0
                                                                    })
                        if not i % PRINT_FREQ:
                            print('batch: ', i, ', loss of discriminator: ', D_loss_curr)

                    # Fix D, train G
                    else:
                        _, G_loss_curr, fm_loss_curr = self.sess.run([self.GC_solver, self.G_loss, self.fm_loss],
                                                            feed_dict={
                                                                self.features: inputdata,
                                                                self.label: y_real_mb,
                                                                self.Z: sample_Z(GAN_BATCH_SIZE, noise_dim),
                                                                self.y_gen: y_fake_mb,
                                                                self.keep_prob: 1.0
                                                            })
                        if not i % PRINT_FREQ:
                            print('batch: ', i, ', loss of generator: ', G_loss_curr)

            print('Training finished')

    def train_save(self):
        self.train()
        self.save(LSTM_AE_GAN_WEIGHTS_DIR, self.theta_D)

if __name__ == '__main__':
    #from autoencoder import Autoencoder, Autoencoder_LSTM
    #Ae = Autoencoder_LSTM()
    #Ae.train_save(save_encoding=True)
    ocan = LSTM_AE_OCAN(pre_encoding=False)
    ocan.train_save()
