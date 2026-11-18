"""Choose a batch by measured throughput only, before GLMM training/validation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','1')
import argparse
import gc
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from sbi_model import atomic_json
from sbi_networks import make_models
from sbi_pilot import configure,gradient_step


def main(output):
    devices=configure();results=[]
    for batch in (128,256,512,1024):
        tf.keras.backend.clear_session();gc.collect()
        npe,cls=make_models(20261118)
        opt1=tf.keras.optimizers.Adam(1e-4);opt2=tf.keras.optimizers.Adam(1e-4)
        x=tf.zeros((batch,300,60));theta=tf.zeros((batch,11));u=tf.zeros((batch,300),tf.int32)
        @tf.function
        def update():
            with tf.GradientTape() as tape:loss=tf.reduce_mean(npe.nll(theta,x,True))
            gradient_step(npe,opt1,loss,tape)
            with tf.GradientTape() as tape:
                loss2=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=u,logits=cls((x,theta),True)))
            gradient_step(cls,opt2,loss2,tape)
            return loss+loss2
        try:
            for _ in range(5):float(update())
            tf.config.experimental.reset_memory_stats('GPU:0');start=time.monotonic()
            for _ in range(20):float(update())
            elapsed=time.monotonic()-start;memory=tf.config.experimental.get_memory_info('GPU:0')
            results.append(dict(batch=batch,steps=20,seconds=elapsed,examples_per_second=20*batch/elapsed,
                                gpu_memory=memory))
            print(results[-1],flush=True)
        except tf.errors.ResourceExhaustedError:
            results.append(dict(batch=batch,error='GPU memory exhausted'));break
        del update,npe,cls,opt1,opt2,x,theta,u
    valid=[r for r in results if 'error' not in r]
    selected=max(valid,key=lambda r:r['examples_per_second'])['batch']
    atomic_json(output,dict(devices=devices,benchmark='Disposable networks and synthetic shape-only inputs; no statistical model selected',
                            results=results,selected_batch=selected))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)
