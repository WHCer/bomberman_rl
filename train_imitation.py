import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import argparse
import pickle
from sklearn.metrics import explained_variance_score, accuracy_score
import matplotlib.pyplot as plt
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
from keras.callbacks import TensorBoard
def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights

    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.

        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
        corresponds to the fan-in, so this makes the initialization usable for
        both dense and convolutional layers.

        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init
def conv_to_fc(input_tensor):
    """
    Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network

    :param input_tensor: (TensorFlow Tensor) The convolutional input tensor
    :return: (TensorFlow Tensor) The fully connected output tensor
    """
    n_hidden = np.prod([v.value for v in input_tensor.get_shape()[1:]])
    input_tensor = tf.reshape(input_tensor, [-1, n_hidden])
    return input_tensor
def load_data(path,index=(0,1,3)):
    obervation = []
    p = []
    v = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            matrix = np.load(os.path.join(dirpath,file))[:, index]
            obervation.append(matrix[:, 0])
            p.append(matrix[:, 1])
            v.append(matrix[:, 2])

    obervation = np.concatenate(obervation)
    obervation = np.stack([x for x in obervation])

    p = np.concatenate(p)
    p = np.stack([x for x in p])

    v = np.concatenate(v)
    v = np.stack([x for x in v])

    unique_obervation,index = np.unique(obervation,return_index=True,axis=0)
    unique_p = p[index]
    unique_v = v[index]

    return unique_obervation,unique_p,unique_v

import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--b', type=int, default=512)
    parser.add_argument('--weight', type=float, default=10)
    args = parser.parse_args()

    training = False
    gamma = 0.99
    phase = 'coins'
    path = '../memory/{}/train'.format(phase)
    value_weight = args.weight
    epochs = args.epochs
    batch_size = args.b

    c = x = Input(shape=(17, 17, 1))
    c = Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', kernel_initializer=ortho_init(np.sqrt(2)),
               data_format='channels_last', bias_initializer=tf.constant_initializer(0.0), activation='relu',
               name='c1')(c)
    c = Conv2D(filters=256, kernel_size=4, strides=1, padding='valid', kernel_initializer=ortho_init(np.sqrt(2)),
               data_format='channels_last', bias_initializer=tf.constant_initializer(0.0), activation='relu',
               name='c2')(c)
    c = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', kernel_initializer=ortho_init(np.sqrt(2)),
               data_format='channels_last', bias_initializer=tf.constant_initializer(0.0), activation='relu',
               name='c3')(c)
    h = Flatten()(c)
    hidden = Dense(units=512, kernel_initializer=ortho_init(np.sqrt(2)),
                   bias_initializer=tf.constant_initializer(0.0),
                   activation='relu', name='f1')(h)
    v = Dense(units=1, kernel_initializer=ortho_init(np.sqrt(1.0)), bias_initializer=tf.constant_initializer(0.0),
              name='vf', activation='tanh')(hidden)
    p = Dense(units=6, kernel_initializer=ortho_init(np.sqrt(1.0)), bias_initializer=tf.constant_initializer(0.0),
              name='pi', activation='softmax')(hidden)
    model = Model(x, [v, p])
    model.summary()

    if not training:
        model.load_weights('policy_{}_disc0.99_weight10_best.h5'.format(phase))
        weight_file = []
        for layer in model.layers: # c1,c2,c3
            if layer.name in ['c1', 'c2', 'c3', 'c4']:
                weights = layer.get_weights()
                weight_file.append(weights[0])
                weight_file.append(weights[1][np.newaxis,:,np.newaxis,np.newaxis])
            if layer.name in ['f1','vf','pi']:
                weights = layer.get_weights()
                weight_file.append(weights[0])
                weight_file.append(weights[1])
        with open('policy_{}_disc0.99_weight10_best.weight'.format(phase), 'wb') as fp:
            pickle.dump(weight_file, fp)
        with open('policy_{}_disc0.99_weight10_best.weight'.format(phase), 'rb') as fp:
            weight_file = pickle.load(fp)

        for i in range(len(weight_file)):
            print(weight_file[i].shape)
    else:
        x_all,p_all,v_all = load_data(path)
        x_train = x_all[:80000]
        p_train = p_all[:80000]
        v_train = v_all[:80000]
        x_test = x_all[80000:]
        p_test = p_all[80000:]
        v_test = v_all[80000:]

        print(x_train.shape,p_train.shape,v_train.shape)
        print(x_test.shape,p_test.shape,v_test.shape)
        # optimizer = Adam(lr=2.5e-4,epsilon=1e-5)
        model.compile(optimizer='adam', loss=['mse','sparse_categorical_crossentropy'], loss_weights=[value_weight, 1], metrics={'pi': 'accuracy'})
        tbCallBack = TensorBoard(log_dir='./coins',  # log 目录
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=False,  # 是否可视化梯度直方图
                                 write_images=False,  # 是否可视化参数
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)
        callbacks = [
            ModelCheckpoint('policy_{}_disc{}_weight{}_best.h5'.format(phase,gamma,value_weight),
                            monitor='val_loss', verbose=1, save_best_only=True, mode='auto',
                            save_weights_only=True),
            EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='auto'),
            tbCallBack
        ]
        print('policy_{}_disc{}_weight{}'.format(phase,gamma,value_weight))
        history = model.fit(x_train, [v_train, p_train], batch_size=batch_size, epochs=epochs, validation_data=(x_test, [v_test, p_test]), callbacks=callbacks)


        model.load_weights('policy_{}_disc{}_weight{}_best.h5'.format(phase,gamma,value_weight))
        v_train_pred,p_train_pred = model.predict(x_train, batch_size=512)
        v_test_pred,p_test_pred = model.predict(x_test, batch_size=512)
        act_train_pred = np.argmax(p_train_pred, axis=1)
        act_test_pred = np.argmax(p_test_pred, axis=1)
        print("Accuracy train:", accuracy_score(p_train, act_train_pred))
        print("Accuracy test:", accuracy_score(p_test, act_test_pred))
        print("Explained variance train:", explained_variance_score(v_train, v_train_pred))
        print("Explained variance test:", explained_variance_score(v_test, v_test_pred))
