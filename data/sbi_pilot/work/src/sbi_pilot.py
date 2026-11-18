"""Bounded, checkpointed SBI pilot. Does not read the original formal test data."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from sbi_model import features,atomic_json,PRIOR,DESIGN
from sbi_networks import Posterior,make_models


def configure():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    devices=tf.config.list_physical_devices('GPU')
    if not devices:raise RuntimeError('CUDA GPU unavailable; do not silently train on CPU')
    for gpu in devices:tf.config.experimental.set_memory_growth(gpu,True)
    return [dict(name=g.name,details=tf.config.experimental.get_device_details(g)) for g in devices]


def load_split(root,split):
    root=Path(root)/split;manifest=json.loads((root/'manifest.json').read_text())
    result={k:[] for k in ('y','u','theta')}
    for name,sha in manifest['files'].items():
        path=root/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,name
        with np.load(path) as data:
            for k in result:result[k].append(data[k])
    return {k:np.concatenate(v) for k,v in result.items()},manifest


def gradient_step(model,optimizer,loss,tape):
    gradients=tape.gradient(loss,model.trainable_variables)
    if any(g is None for g in gradients):raise RuntimeError('Missing neural gradient')
    tf.debugging.assert_all_finite(loss,'Non-finite loss')
    gradients,norm=tf.clip_by_global_norm(gradients,10.)
    tf.debugging.assert_all_finite(norm,'Non-finite gradient')
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return norm


def analytic_toy(output,steps=2000):
    """Gaussian conjugate posterior checks flow direction, conditioning and reload."""
    tf.keras.utils.set_random_seed(20261718)
    rng=np.random.default_rng(20261718);net=Posterior(dim=2,summary=False,layers=4)
    net((tf.zeros((2,2)),tf.zeros((2,2))))
    opt=tf.keras.optimizers.Adam(learning_rate=8e-4)
    @tf.function(reduce_retracing=True)
    def update(theta,x):
        with tf.GradientTape() as tape:loss=tf.reduce_mean(net.nll(theta,x,True))
        gradient_step(net,opt,loss,tape)
        return loss
    begin=time.monotonic();losses=[]
    for step in range(steps):
        theta=rng.normal(size=(256,2)).astype(np.float32)
        x=theta+rng.normal(size=theta.shape).astype(np.float32)/np.sqrt(20.)
        loss=float(update(theta,x))
        if (step+1)%250==0:
            losses.append(dict(step=step+1,nll=loss));print('toy',losses[-1],flush=True)
    theta=rng.normal(size=(200,2));x=(theta+rng.normal(size=theta.shape)/np.sqrt(20.)).astype(np.float32)
    base=rng.normal(size=(200,1024,2)).astype(np.float32)
    draws=net.sample(x,base).numpy();mean=x*20/21;sd=1/np.sqrt(21.)
    rmse=float(np.sqrt(np.mean((draws.mean(1)-mean)**2))/sd)
    variance=float(np.median(draws.var(1,ddof=1)/(sd*sd)))
    interval=np.quantile(draws,[.025,.975],axis=1)
    coverage=float(((theta>=interval[0])&(theta<=interval[1])).mean())
    z,ld=net((tf.convert_to_tensor(theta,dtype=tf.float32),x))
    reverse=net.flow(z,x,inverse=True,training=False).numpy()
    roundtrip=float(abs(reverse-theta).max())
    probe=draws[:4,:32].copy();ckpt=tf.train.Checkpoint(model=net,optimizer=opt)
    prefix=ckpt.save(str(output/'toy_state'))
    tf.keras.utils.set_random_seed(20261718)
    reloaded=Posterior(dim=2,summary=False,layers=4);reloaded((tf.zeros((2,2)),tf.zeros((2,2))))
    tf.train.Checkpoint(model=reloaded).restore(prefix).expect_partial()
    reload_error=float(abs(probe-reloaded.sample(x[:4],base[:4,:32]).numpy()).max())
    passed=rmse<.5 and .7<variance<1.3 and .89<coverage<.99 and roundtrip<1e-4 and reload_error<1e-5
    result=dict(steps=steps,seed=20261718,seconds=time.monotonic()-begin,
                posterior_mean_RMSE_in_exact_SD=rmse,median_variance_ratio=variance,
                coverage_95=coverage,flow_roundtrip_max_error=roundtrip,
                reload_draw_max_error=reload_error,thresholds=dict(mean_RMSE_SD=.5,
                median_variance_ratio=[.7,1.3],coverage=[.89,.99]),passed=bool(passed),losses=losses)
    atomic_json(output/'toy_check.json',result);print(json.dumps(result,indent=2),flush=True)
    if not passed:raise RuntimeError('Analytic toy check failed; do not train GLMM')


def save_state(output,step,checkpoint,history,configuration):
    final=output/f'checkpoint_{step:06d}';temporary=output/f'checkpoint_{step:06d}.tmp'
    temporary.mkdir(exist_ok=False)
    checkpoint.write(str(temporary/'state'))
    atomic_json(temporary/'metadata.json',dict(step=step,configuration=configuration,
                                             deterministic_batch_seed=[configuration['training_seed'],step+1]))
    temporary.replace(final)
    atomic_json(output/'history.json',history)
    atomic_json(output/'progress.json',dict(step=step,checkpoint=final.name,latest=history[-1]))


def train(args,gpus):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    config=dict(stage='pilot, not formal comparison',prior=PRIOR,design=DESIGN,
                framework='BayesFlow 1.1.6; TF 2.15.1, paper-compatible legacy implementation',
                network_seed=20261118,training_seed=20261818,max_steps=args.steps,
                max_seconds=args.minutes*60.,batch_size=args.batch,validation_every=500,
                early_stop_patience=8,prior_statistics_seed=20261918,
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (Path(__file__),Path(__file__).with_name('sbi_networks.py'),Path(__file__).with_name('sbi_model.py'))},
                gpu=gpus,unique_training_datasets=100000,training_mode='offline, sampling with replacement')
    previous=None
    if (output/'configuration.json').exists():
        if not args.resume:raise FileExistsError('Use a new run directory or --resume')
        previous=json.loads((output/'configuration.json').read_text())
        for key in ('prior','design','sources','batch_size','max_steps','max_seconds','training_seed','network_seed'):
            assert previous[key]==config[key],key
    train_data,train_manifest=load_split(args.data,'train');val,_=load_split(args.data,'validation')
    config['unique_training_datasets']=len(train_data['theta'])
    assert train_manifest['spec']['prior']==PRIOR and train_manifest['spec']['design']==DESIGN
    assert train_manifest['spec']['simulator_sha256']==config['sources']['sbi_model.py']
    atomic_json(output/'configuration.json',config)
    # Fixed preprocessing from training targets, no validation/calibration/test fitting.
    mean=train_data['theta'].mean(0,dtype=np.float64).astype(np.float32)
    scale=train_data['theta'].std(0,dtype=np.float64).astype(np.float32)
    assert np.min(scale)>.01
    atomic_json(output/'standardization.json',dict(mean=mean.tolist(),scale=scale.tolist(),from_split='train'))
    train_theta=(train_data['theta']-mean)/scale
    val_theta=(val['theta']-mean)/scale;val_features=features(val['y'])
    posterior,classifier=make_models(config['network_seed'])
    # Verify person-level permutation invariance before fitting, not time-point shuffling.
    probe=val_features[:4];permutation=np.random.default_rng(20262118).permutation(300)
    e1=posterior.condition(probe).numpy();e2=posterior.condition(probe[:,permutation]).numpy()
    invariant_error=float(abs(e1-e2).max());assert invariant_error<1e-4
    atomic_json(output/'network_check.json',dict(subject_permutation_max_error=invariant_error,
                                               prior_and_simulator_hash_verified=True))
    lr=tf.keras.optimizers.schedules.CosineDecay(5e-4,args.steps,alpha=.2)
    opt_npe=tf.keras.optimizers.Adam(lr);opt_cls=tf.keras.optimizers.Adam(lr)
    opt_npe.build(posterior.trainable_variables);opt_cls.build(classifier.trainable_variables)
    step_var=tf.Variable(0,dtype=tf.int64,trainable=False)
    checkpoint=tf.train.Checkpoint(posterior=posterior,classifier=classifier,
                                  optimizer_npe=opt_npe,optimizer_classifier=opt_cls,step=step_var)
    @tf.function(reduce_retracing=True)
    def update(x,theta,u):
        with tf.GradientTape() as tape:
            npe=tf.reduce_mean(posterior.nll(theta,x,True))
        g1=gradient_step(posterior,opt_npe,npe,tape)
        with tf.GradientTape() as tape:
            logits=classifier((x,theta),True)
            cls=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=u,logits=logits))
        g2=gradient_step(classifier,opt_cls,cls,tape)
        return npe,cls,g1,g2
    @tf.function(reduce_retracing=True)
    def validation(x,theta,u):
        npe=tf.reduce_mean(posterior.nll(theta,x,False))
        logits=classifier((x,theta),False)
        cls=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=u,logits=logits))
        return npe,cls
    start_step=0;elapsed_before=0.
    history=[];best_npe=best_cls=float('inf');best_npe_step=best_cls_step=0;last_change=0
    if previous:
        progress=json.loads((output/'progress.json').read_text());start_step=progress['step']
        checkpoint.read(str(output/progress['checkpoint']/'state')).assert_existing_objects_matched()
        history=[v for v in json.loads((output/'history.json').read_text()) if v['step']<=start_step]
        elapsed_before=history[-1]['seconds']
        last_change=int(np.argmin([v['validation_nll'] for v in history]))
        best_npe=history[last_change]['validation_nll'];best_npe_step=history[last_change]['step']
        j=int(np.argmin([v['validation_ce'] for v in history]));best_cls=history[j]['validation_ce'];best_cls_step=history[j]['step']
    def batch_generator():
        for step in range(start_step+1,args.steps+1):
            rng=np.random.default_rng(np.random.SeedSequence([config['training_seed'],step]))
            indices=rng.integers(len(train_theta),size=args.batch)
            yield features(train_data['y'][indices]),train_theta[indices],train_data['u'][indices].astype(np.int32)
    signature=(tf.TensorSpec((args.batch,300,60),tf.float32),
               tf.TensorSpec((args.batch,11),tf.float32),tf.TensorSpec((args.batch,300),tf.int32))
    dataset=tf.data.Dataset.from_generator(batch_generator,output_signature=signature).prefetch(4)
    begin=time.monotonic();window=time.monotonic();running=[];reason='step_budget'
    step=start_step
    for step,batch in enumerate(dataset,start_step+1):
        values=[float(v) for v in update(*batch)];running.append(values[:2]);step_var.assign(step)
        if step%50==0:
            atomic_json(output/'live_progress.json',dict(step=step,training_nll=values[0],training_ce=values[1],
                                                        seconds=elapsed_before+time.monotonic()-begin))
        if step%500==0 or step==args.steps or elapsed_before+time.monotonic()-begin>args.minutes*60:
            val_losses=[]
            for j in range(0,len(val_theta),64):
                val_losses.append([float(v) for v in validation(val_features[j:j+64],val_theta[j:j+64],
                                                               val['u'][j:j+64].astype(np.int32))])
            vn,vc=np.mean(val_losses,axis=0)
            if vn<best_npe-1e-4:best_npe=float(vn);best_npe_step=step;last_change=len(history)
            if vc<best_cls:best_cls=float(vc);best_cls_step=step
            item=dict(step=step,examples_seen=step*args.batch,seconds=elapsed_before+time.monotonic()-begin,
                      train_nll=float(np.mean(running,axis=0)[0]),train_ce=float(np.mean(running,axis=0)[1]),
                      validation_nll=float(vn),validation_ce=float(vc),
                      examples_per_second=len(running)*args.batch/(time.monotonic()-window),
                      gpu_memory=tf.config.experimental.get_memory_info('GPU:0'))
            history.append(item);save_state(output,step,checkpoint,history,config)
            atomic_json(output/'selection.json',dict(best_npe_step=best_npe_step,best_classifier_step=best_cls_step,
                                                     best_validation_nll=best_npe,best_validation_ce=best_cls))
            print(json.dumps(item),flush=True);running=[];window=time.monotonic()
            if len(history)-1-last_change>=config['early_stop_patience']:
                reason='validation_patience';break
        if elapsed_before+time.monotonic()-begin>args.minutes*60:
            reason='wall_time_budget';break
    atomic_json(output/'training_complete.json',dict(reason=reason,steps=step,seconds=elapsed_before+time.monotonic()-begin,
                                                    best_npe_step=best_npe_step,best_classifier_step=best_cls_step,
                                                    calibration_not_used_for_selection=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=10000);ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--minutes',type=float,default=90.);ap.add_argument('--toy-only',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);gpu=configure()
    if not args.resume:analytic_toy(args.output)
    else:assert json.loads((args.output/'toy_check.json').read_text())['passed']
    if not args.toy_only:train(args,gpu)
