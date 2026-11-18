"""Paper-standard conditional normalizing flow and separate conditional classifier.

BayesFlow 1.1.6 / TensorFlow 2.15.1; uses the paper's NPE and classification
losses without changing the GLMM or adding an EM update. Fixed T=20 only.
"""
import numpy as np
import tensorflow as tf
from bayesflow.inference_networks import InvertibleNetwork


def dense(units):
    return tf.keras.layers.Dense(units,activation='elu')


class Summary(tf.keras.Model):
    """DeepSet over independent people; each row retains the entire trajectory."""
    def __init__(self):
        super().__init__()
        self.inner=tf.keras.Sequential([dense(128),dense(128),dense(64)])
        self.outer=tf.keras.Sequential([dense(128),dense(128),tf.keras.layers.Dense(64)])

    def call(self,x,training=False):
        return self.outer(tf.reduce_mean(self.inner(x,training=training),axis=1),training=training)


class Posterior(tf.keras.Model):
    def __init__(self,dim=11,summary=True,layers=6):
        super().__init__()
        self.encoder=Summary() if summary else None
        self.flow=InvertibleNetwork(num_params=dim,num_coupling_layers=layers,
                                   coupling_design='affine',use_act_norm=True,
                                   coupling_settings=dict(dense_args=dict(units=128,activation='elu',kernel_regularizer=None),
                                                          num_dense=2,dropout=False,mc_dropout=False))

    def condition(self,x,training=False):
        return self.encoder(x,training=training) if self.encoder is not None else x

    def call(self,inputs,training=False):
        theta,x=inputs
        return self.flow(theta,self.condition(x,training),training=training)

    def nll(self,theta,x,training=False):
        z,logdet=self((theta,x),training=training)
        return .5*tf.reduce_sum(z*z+np.log(2*np.pi),axis=-1)-logdet

    def sample(self,x,base):
        """base: (datasets, draws, dim); no hidden data or true theta in x."""
        c=self.condition(x,False)
        c=tf.broadcast_to(c[:,None,:],tf.concat((tf.shape(base)[:2],[tf.shape(c)[-1]]),axis=0))
        return self.flow(base,c,inverse=True,training=False)


class Classifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.local=tf.keras.Sequential([dense(128),dense(128),dense(32)])
        self.net=tf.keras.Sequential([dense(128),dense(128),tf.keras.layers.Dense(3)])

    def call(self,inputs,training=False):
        x,theta=inputs
        local=self.local(x,training=training)
        theta=tf.broadcast_to(theta[:,None,:],(tf.shape(x)[0],tf.shape(x)[1],tf.shape(theta)[-1]))
        return self.net(tf.concat((local,theta),axis=-1),training=training)


def make_models(seed=20261118):
    tf.keras.utils.set_random_seed(seed)
    posterior=Posterior();classifier=Classifier()
    x=tf.zeros((2,300,60));theta=tf.zeros((2,11))
    posterior((theta,x));classifier((x,theta))
    return posterior,classifier
