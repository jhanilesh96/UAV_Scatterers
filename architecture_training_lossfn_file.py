# The file is not standlone and only provided to give an idea of the architecture, training and loss function used in the paper 
# "Temporally Correlated Compressed sensing using Generative Models for Channel Estimation in Unmanned Aerial Vehicles, IEEE Transactions on Wireless Communications"

import numpy as np
import scipy.constants as constants
import sys
import time
import myutils

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp

import c_commons as commons


N, M_x, M_y, M_x_ft, M_y_ft,frames = \
    commons.N, commons.M_x, commons.M_y, commons.DFTfactor*commons.M_x, commons.DFTfactor*commons.M_y, commons.frames

dim_x = dim_h = M_x_ft * M_y_ft * N;

'''' 0. Configurations '''
if __name__ == '__main__' and len(sys.argv)>1:
    config = int(sys.argv[1]);
else:
    config = commons.c20_default_config

if __name__ == '__main__' and len(sys.argv)>2:
    train_debug = int(sys.argv[2]);
else:
    train_debug = commons.c20_default_train_debug

print('CONFIG :::', config)

sim_dot_after = 1
sim_LOS = True
dim_z = dim_h//2; dim_zo = dim_z;
dim_g = dim_h//4; dim_ho = dim_h;
layers_n1 = [2*dim_z, 2*dim_z] # 1.1.1 Phi_prior ::: Dense (stochastic) ::: z_t | g_t_1 => dim_g -> 2(dim_z)
layers_nz = [dim_z] # 1.1.2.A Phi_z ::: Dense (function) ::: z_t -> zo_t => dim_z -> dim_zo
filters_n2 = [25,20,10] # 1.1.2 Phi_dec ::: ConvT (stochastic) ::: h_t | Phi_z(z_t), g_t_1 => dim_zo + dim_g -> 2(dim_h)
filters_nh = [20,10,1] # 1.1.3.A Phi_h ::: Conv (function) ::: h_t -> ho_t => dim_h -> dim_ho
layers_n3 = [2*dim_g, 1.5*dim_g, dim_g] # 1.1.4 f_theta ::: Dense (function) ::: g_t | g_t_1, Phi_h(h_t), Phi_z(z_t) => dim_ho + dim_zo + dim_g -> dim_g
layers_n4 = [2*dim_z, 2*dim_z] # 1.2.1 Phi_enc ::: Dense (stochastic) ::: z_t | g_t_1, Phi_h(h_t) => dim_g + dim_ho -> dim_z
sim_epochs = 1000;
sim_batch_size = 25;
sim_loadDataConfig = 1
train_files_override = -1
sim_lr = 3e-4; sim_clipgrad = 1e-1;
sim_decay_epoch_lr = 1 # decay lr after every sim_decay_epoch_lr 
sim_decay_lr = 0.01 # by 10**-sim_decay_lr
sim_min_lr = 2e-6
sim_restore = True
sim_base_epoch_kl = 10
sim_decay_kl = 0.01 # scale_kl = 1-10**(-sim_decay_kl*(sim_base_epoch_kl+_epoch));  e**-7 = 1e-3, e**-10 = 1e-5
sim_base_epoch_ll = 50
sim_decay_ll = 1.0 # scale_ll = 1-10**(-sim_decay_ll*(sim_base_epoch_ll+_epoch));  e**-7 = 1e-3, e**-10 = 1e-5
sim_base_epoch_rate = 0
sim_decay_rate = 0.01 # dropout_rate_gist = 10**(-sim_decay_rate*(sim_base_epoch_rate+_epoch)).astype(commons.ftype);  e**-7 = 1e-3, e**-10 = 1e-5

if config==0 or config==-1: # for test and error -520
    pass;

# if config==1:
#     sim_LOS = False

# if config==1:
#     sim_loadDataConfig = 4
#     sim_epochs=500
#     sim_base_epoch_rate = 1
#     sim_decay_rate = 0.05
#     sim_decay_kl = 0.05
#     filters_n2 = [25,20,20] 
#     sim_decay_lr = 0.01
#     assert N == 8


# if config==3:
#     sim_LOS = False
#     dim_z = int(0.25*dim_h); dim_zo = dim_z;
#     sim_loadDataConfig = 4
#     sim_epochs=125
#     sim_base_epoch_rate = 1
#     sim_decay_rate = 0.5
#     sim_decay_kl = 0.2
#     filters_n2 = [50,32,16] 
#     sim_decay_lr = 0.01
#     assert N == 16

''' NLOS low K '''
if config==66:
    sim_LOS = False
    dim_z = int(0.10*dim_h); dim_zo = dim_z;
    sim_loadDataConfig = 6
    sim_epochs=125
    sim_base_epoch_rate = 1
    sim_decay_rate = 0.5
    sim_decay_kl = 0.001
    filters_n2 = [48,32,16] 
    sim_decay_lr = 0.01
    assert N == 16

''' NLOS moderate K '''
if config==6:
    sim_LOS = False
    dim_z = int(0.4*dim_h); dim_zo = dim_z;
    sim_loadDataConfig = 6
    sim_epochs=500
    sim_base_epoch_rate = 1
    sim_decay_rate = 10.0
    sim_base_epoch_kl = 1
    sim_decay_kl = 0.01
    filters_n2 = [40,20,10] 
    sim_decay_lr = 0.01
    assert N == 16

''' NLOS full K '''
if config==5:
    sim_LOS = False
    dim_z = int(dim_h); dim_zo = dim_z;
    sim_loadDataConfig = 6
    sim_epochs=500
    sim_base_epoch_rate = 1
    sim_decay_rate = 0.05
    sim_decay_kl = 0.05
    filters_n2 = [2,2,2] 
    sim_decay_lr = 0.01
    assert N == 16

''' NLOS low K '''
if config==86:
    sim_LOS = False
    dim_z = int(0.25*dim_h); dim_zo = dim_z;
    sim_loadDataConfig = 8
    sim_epochs=125
    sim_base_epoch_rate = 1
    sim_decay_rate = 0.5
    sim_decay_kl = 0.2
    filters_n2 = [8,16,32,64]
    sim_decay_lr = 0.01
    sim_lr = 1e-4;
    sim_min_lr = 1e-7
    assert N == 32



''' LOS moderate K '''
if config==7:
    sim_restore = True
    sim_LOS = True
    sim_lr = 1e-5;
    dim_z = int(0.25*dim_h); dim_zo = dim_z;
    sim_loadDataConfig = 6
    sim_epochs=125
    sim_base_epoch_rate = 10000
    sim_base_epoch_kl = 10000
    filters_n2 = [32,16,8] 
    assert N == 16


''' LOS low K - works great '''
if config==67:
    sim_LOS = True
    dim_z = int(0.10*dim_h); dim_zo = dim_z;
    sim_loadDataConfig = 6
    sim_epochs=125
    sim_base_epoch_rate = 1
    sim_decay_rate = 0.5
    sim_decay_kl = 0.2
    filters_n2 = [48,32,16] 
    sim_decay_lr = 0.01
    assert N == 16


'''' 1. Networks '''
'''' 1.1  Generative Networks '''
'''' 1.1.1 Phi_prior ::: Dense (stochastic) ::: z_t | g_t_1 => dim_g -> 2(dim_z) '''
i = tfk.layers.Input(shape=dim_g); _ = i
for _num in layers_n1:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.Dense(dim_z + dim_z)(_)
Phi_prior = tfk.Model(inputs=i, outputs=_, name='Phi_prior')
Phi_prior.summary()

'''' 1.1.2.A Phi_z ::: Dense (function) ::: z_t -> zo_t => dim_z -> dim_zo '''
i = tfk.layers.Input(shape=dim_z); _ = i
for _num in layers_nz:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.Dense(dim_zo)(_)
Phi_z = tfk.Model(inputs=i, outputs=_, name='Phi_z')
Phi_z.summary()

'''' 1.1.2 Phi_dec ::: ConvT (stochastic) ::: h_t | Phi_z(z_t), g_t_1 => dim_zo + dim_g -> 2(dim_h) '''
i = tf.keras.layers.Input(shape=dim_zo+dim_g);
_ = tf.keras.layers.Dense(units=N*M_x_ft*M_y_ft*filters_n2[0], activation='relu')(i)
_ = tf.keras.layers.Reshape(target_shape=(N, M_x_ft*M_y_ft, filters_n2[0]))(_)
for _filters in filters_n2[1:]:
    _ = tf.keras.layers.Conv2DTranspose(filters=_filters, kernel_size=3, strides=1, padding='same', activation='relu')(_)
_ = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=3, strides=1, padding='same')(_)
Phi_dec = tfk.Model(inputs=i, outputs=_, name='Phi_dec')
Phi_dec.summary()

'''' 1.1.3.A Phi_h ::: Conv (function) ::: h_t -> ho_t => dim_h -> dim_ho '''
i = tfk.layers.Input(shape=(N,M_x_ft*M_y_ft,1)); _ = i
for _filters in filters_nh:
    _ = tfk.layers.Conv2D(filters=_filters, padding='same', kernel_size=3, strides=(1, 1), activation='relu')(_)
_ = tfk.layers.Flatten()(_)
_ = tfk.layers.Dense(dim_ho)(_)
Phi_h = tfk.Model(inputs=i, outputs=_, name='Phi_h')
Phi_h.summary()

'''' 1.1.4 f_theta ::: Dense (function) ::: g_t | g_t_1, Phi_h(h_t), Phi_z(z_t) => dim_ho + dim_zo + dim_g -> dim_g '''
i = tfk.layers.Input(shape=dim_ho+dim_zo+dim_g); _ = i
for _num in layers_n3:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.Dense(dim_g)(_)
f_theta = tfk.Model(inputs=i, outputs=_, name='f_theta')
f_theta.summary()


''''1.2 Inference Networks '''
'''' 1.2.1 Phi_enc ::: Dense (stochastic) ::: z_t | g_t_1, Phi_h(h_t) => dim_g + dim_ho -> dim_z '''
i = tfk.layers.Input(shape=dim_ho+dim_g); _ = i
for _num in layers_n4:
    _ = tfk.layers.Dense(int(_num), activation='relu')(_)
_ = tfk.layers.Dense(dim_z + dim_z)(_)
Phi_enc = tfk.Model(inputs=i, outputs=_, name='Phi_enc')
Phi_enc.summary()




'''' Helper Functions '''
def dot(_type=1):
    if _type==0:
        s = '%'
    elif _type==1:
        s = '.'
    elif _type==2:
        s = '!'
    elif _type==3:
        s = '$'
    print(s,end='',flush=True);


'''' 2. Combined Network '''
trainable_variables = []
mlps = [Phi_prior, Phi_z, Phi_dec, Phi_h, f_theta, Phi_enc]
for _mlp in mlps:
    for _trainable_variables in _mlp.trainable_variables:
        trainable_variables.append(_trainable_variables);

print('Dimensions Matrix:{0:}, h:{1:}, z:{2:}, g:{3:}'.format(N*M_x_ft*M_y_ft, dim_h, dim_z, dim_g))
print('Number of Trainable variables : ', np.sum([tf.size(_) for _ in trainable_variables]))

# train_opt = tfk.optimizers.SGD(learning_rate=sim_lr, momentum=0.01, nesterov=True)
train_opt = tfk.optimizers.Adam(learning_rate=sim_lr)
train_dot_after_points = 20
train_checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=train_opt, net=mlps)
train_folder = './results_big/training_checkpoints/train'+str(config)
train_manager = tf.train.CheckpointManager(train_checkpoint, train_folder, max_to_keep=10)
train_baseEpoch = 0;

for _i in range(20000):
    if myutils.check('channels_s__'+str(_i), folderName='data', subFolderName='data'+str(sim_loadDataConfig)):
        pass;
    else:
        train_files = _i
        break

if train_files==0:
    for _i in range(20000):
        if myutils.check('channels_NLOS_s__'+str(_i), folderName='data', subFolderName='data'+str(sim_loadDataConfig)):
            pass;
        else:
            train_files = _i
            break
    if train_files==0:
        print("cant seem to find training files")
        exit()

print("train_files", train_files)

m  = train_files if (train_files_override < 0 or train_files<train_files_override) else train_files_override
if train_files>10:
    train_files = train_files-15;

def savefigs_true():
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    samples = channels_s[np.random.permutation(channels_s.shape[0])[:10]]; samples = np.transpose(samples, [1,0,2,3,4]);
    plt.figure(figsize=[19.2,12]);
    plt.subplot(len(samples), samples[0].shape[0], samples[0].shape[0]*len(samples))
    for _frame in range(len(samples)):
        for _sample in range(samples[0].shape[0]):
            _ = plt.subplot(len(samples), samples[0].shape[0], _frame*samples[0].shape[0]+_sample+1)
            _ = plt.imshow(np.abs(samples[_frame][_sample]).reshape([N,M_x_ft*M_y_ft]))
            _ = plt.ylabel(str(_sample)+'_'+str(_frame))

    plt.savefig('./results_big/Logs/train'+str(config)+'/true_samples')

if sim_restore and train_manager.latest_checkpoint is not None:
    channels_s = myutils.load(('channels_s__' if sim_LOS else 'channels_NLOS_s__')+str(0), folderName='data', subFolderName='data'+str(sim_loadDataConfig))
    status = train_checkpoint.restore(train_manager.latest_checkpoint)
    print("\n\nWEIGHTS ::: Restored from {}".format(train_manager.latest_checkpoint),'\n\n')
    train_baseEpoch = int(train_manager.latest_checkpoint.split('-')[-1])-1
    [logg_losses, logg_ll, logg_kl] = myutils.load('logg_losses', folderName='results_big', subFolderName='Logs/train'+str(config))
    num_samples = logg_losses.shape[2]
    if sim_epochs>logg_losses.shape[0]:
        logg_losses = np.concatenate([logg_losses,\
             np.zeros(shape=[sim_epochs-logg_losses.shape[0], train_files, num_samples], dtype='float32')], axis=0)
        logg_ll = np.concatenate([logg_ll,\
             np.zeros(shape=[sim_epochs-logg_ll.shape[0], train_files, num_samples], dtype='float32')], axis=0)
        logg_kl = np.concatenate([logg_kl,\
             np.zeros(shape=[sim_epochs-logg_kl.shape[0], train_files, num_samples], dtype='float32')], axis=0)
        myutils.save('logg_losses', [logg_losses, logg_ll, logg_kl], folderName='results_big', subFolderName='Logs/train'+str(config))
        myutils.save('logg_losses_bak', [logg_losses, logg_ll, logg_kl], folderName='results_big', subFolderName='Logs/train'+str(config))
else:
    print("\n\nWEIGHTS ::: Initializing from scratch.\n\n")
    channels_s = myutils.load(('channels_s__' if sim_LOS else 'channels_NLOS_s__')+str(0), folderName='data', subFolderName='data'+str(sim_loadDataConfig))
    num_samples = channels_s.shape[0]; assert num_samples%sim_batch_size == 0;
    logg_losses = np.zeros(shape=[sim_epochs, train_files, num_samples], dtype='float32')
    logg_ll = np.zeros(shape=[sim_epochs, train_files, num_samples], dtype='float32')
    logg_kl = np.zeros(shape=[sim_epochs, train_files, num_samples], dtype='float32')
    myutils.save('logg_losses', [logg_losses, logg_ll, logg_kl], folderName='results_big', subFolderName='Logs/train'+str(config))
    myutils.save('logg_losses_bak', [logg_losses, logg_ll, logg_kl], folderName='results_big', subFolderName='Logs/train'+str(config))
    savefigs_true();

def checkGrads(_loss, _gradients):
    _gradients_check1 = [np.any(np.isinf(_ if _ is not None else 1)) for _ in _gradients]
    _gradients_check2 = [np.any(np.isnan(_ if _ is not None else 1)) for _ in _gradients]
    if np.isinf(_loss)|np.isnan(_loss)|np.any(np.isinf(_gradients_check1))|np.any(np.isnan(_gradients_check2)):
        dot(0); 
        if train_debug>5:
            print(np.isinf(_loss))
            print(np.isnan(_loss))
            print(np.any(np.isinf(_gradients_check1)))
            print(np.any(np.isnan(_gradients_check2)))
        return None;
    else:
        return _gradients


def savefigs():
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    samples = GenerateSamples();
    plt.figure(figsize=[19.2,12]);
    plt.subplot(len(samples), samples[0].shape[0], samples[0].shape[0]*len(samples))
    for _frame in range(len(samples)):
        for _sample in range(samples[0].shape[0]):
            _ = plt.subplot(len(samples), samples[0].shape[0], _frame*samples[0].shape[0]+_sample+1)
            _ = plt.imshow(samples[_frame][_sample].numpy().reshape([N,M_x_ft*M_y_ft]))
            plt.ylabel(str(_sample)+'_'+str(_frame))
    plt.savefig('./results_big/Logs/train'+str(config)+'/epoch_'+'{:05d}'.format(_epoch))
    plt.savefig('./results_big/Logs/train'+str(config)+'/epoch_current')

    plt.figure(figsize=[4*6.4, 4*4.8]);
    plt.plot((np.arange((_epoch+1)*train_files)+1)/train_files, np.mean(logg_losses[:_epoch+1,:,:],axis=(-1)).ravel())
    plt.plot((np.arange((_epoch+1)*train_files)+1)/train_files, np.mean(logg_ll[:_epoch+1,:,:],axis=(-1)).ravel())
    plt.gca().set_prop_cycle(None)
    plt.plot((np.arange(_epoch+1)*train_files+train_files)/train_files, np.mean(logg_losses[:_epoch+1,:,:],axis=(-1,-2)), '--o' )
    plt.plot((np.arange(_epoch+1)*train_files+train_files)/train_files, np.mean(logg_ll[:_epoch+1,:,:],axis=(-1,-2)), 'o' )
    plt.legend(['OBJ','LL'])
    plt.savefig('./results_big/Logs/train'+str(config)+'/epoch_current_loss')
    plt.close('all')
    
def EndOfEpoch():
    print('\n')
    if (_epoch+1)%sim_decay_epoch_lr == 0:
        dot(3);
        if train_opt.lr > sim_min_lr:
            train_opt.lr = (train_opt.lr*(10**-sim_decay_lr))
        else:
            train_opt.lr = sim_min_lr
    myutils.save('logg_losses', [logg_losses, logg_ll, logg_kl], folderName='results_big', subFolderName='Logs/train'+str(config))
    myutils.save('logg_losses_bak', [logg_losses, logg_ll, logg_kl], folderName='results_big', subFolderName='Logs/train'+str(config))
    train_manager.save();
    savefigs();
    print('Model Saved, OBJ :', np.mean(logg_losses[_epoch]), np.std(logg_losses[_epoch]))
    print('LL :', np.mean(logg_ll[_epoch]), np.std(logg_ll[_epoch]), ', scale :',scale_ll)
    print('KL :', np.mean(logg_kl[_epoch]), np.std(logg_kl[_epoch]), ', scale :',scale_kl)
    print('dropout_rate_gist :', dropout_rate_gist)
    print('lr :', train_opt.lr.numpy(), ', epoch :', str(_epoch)+'/'+str(sim_epochs))
    print('\n')


train_params = [dim_x, dim_z, dim_g, tfk.layers.Dropout]
gen_params = [dim_x, dim_z, dim_g]
ForwardLosses = lambda sample, scale_kl, scale_ll, dropout_rate_gist : commons.ForwardLosses_v2(sample, mlps, params=train_params, scale_kl=scale_kl, scale_ll=scale_ll, dropout_rate_gist=dropout_rate_gist);
GenerateSamples = lambda : commons.GenerateSamples(mlps=mlps, params=gen_params, batches=5, _frames=20)
powpow = lambda x : np.power(10,x)


if config==-1:
    if len(sys.argv)>2:
        _temp = int(sys.argv[2]);
    samples = GenerateSamples()
    channels_s = myutils.load(('channels_s__' if sim_LOS else 'channels_NLOS_s__')+str((5+0)%train_files), folderName='data', subFolderName='data'+str(sim_loadDataConfig))
    train_idxs = np.arange(start=0,stop=num_samples+sim_batch_size,step=sim_batch_size)
    _channel_samples = np.abs(channels_s[train_idxs[0]:train_idxs[0+1]])
    train_folder = './results_big/training_checkpoints/train'+str(_temp)
    train_manager = tf.train.CheckpointManager(train_checkpoint, train_folder, max_to_keep=10)
    status = train_checkpoint.restore(train_manager.latest_checkpoint)
    exit()

do_once = False
def doOnce():
    global do_once
    import matplotlib.pyplot as plt
    if not do_once:
        _ = plt.imshow(_channel_samples[0,:,:N,:,:].reshape([20*N,M_x_ft*M_y_ft]))
        plt.show()
        do_once = True;

if __name__ == '__main__':
    _epoch = -1; 
    if train_baseEpoch==0:
        savefigs();
    train_idxs = np.arange(start=0,stop=num_samples+sim_batch_size,step=sim_batch_size)
    for __epoch in range(sim_epochs):
        _epoch = __epoch + train_baseEpoch
        __time = time.time()
        scale_kl = (1-powpow(-sim_decay_kl*(sim_base_epoch_kl+_epoch))).astype(commons.ftype)
        scale_ll = (1-powpow(-sim_decay_ll*(sim_base_epoch_ll+_epoch))).astype(commons.ftype)
        dropout_rate_gist = powpow(-sim_decay_rate*(sim_base_epoch_rate+_epoch)).astype(commons.ftype)
        for _file in range(train_files):
            _time = time.time()
            channels_s = myutils.load(('channels_s__' if sim_LOS else 'channels_NLOS_s__')+str((5+_file)%train_files), folderName='data', subFolderName='data'+str(sim_loadDataConfig))
            num_samples = channels_s.shape[0]
            for _sample_idx in range(len(train_idxs)-1):
                _channel_samples = np.abs(channels_s[train_idxs[_sample_idx]:train_idxs[_sample_idx+1]])
                _gotThrough = False; #doOnce();
                while not _gotThrough:
                    with tf.GradientTape() as tape:
                        objective, metadata = ForwardLosses(_channel_samples, scale_kl=scale_kl, scale_ll=scale_ll, dropout_rate_gist=dropout_rate_gist)
                        loss = -objective
                    gradients = tape.gradient(loss, trainable_variables)
                    gradients = checkGrads(loss, gradients);
                    _gotThrough = gradients is not None
                    # if not _gotThrough and train_debug>0:
                    #     print('\n',metadata[1].numpy(), metadata[2].numpy)
                train_opt.apply_gradients(
                    (grad, var) 
                    for (grad, var) in zip(gradients, trainable_variables) 
                    if grad is not None
                )
                logg_losses[_epoch, _file, train_idxs[_sample_idx]:train_idxs[_sample_idx+1]] = np.mean(metadata[0], axis=0)
                logg_ll[_epoch, _file, train_idxs[_sample_idx]:train_idxs[_sample_idx+1]] = np.mean(metadata[1], axis=0)
                logg_kl[_epoch, _file, train_idxs[_sample_idx]:train_idxs[_sample_idx+1]] = np.mean(metadata[2], axis=0)
                if train_debug>=3:
                    print('OBJ ::', np.mean(metadata[0]), '\t', np.mean(metadata[1]), '\t',np.mean(metadata[2]))
                if _sample_idx%sim_dot_after==0:
                    dot(1)
            print(_epoch, _file, end='\t', flush=True)
            if train_debug>=2:
                # print('OBJ, LL, KL, time ::', '{0:.2f} \t {1:.2f} \t {2:.2f} \t {3} \t {4}'.format(np.mean(logg_losses[_epoch, _file]), np.mean(logg_ll[_epoch, _file]), np.mean(logg_kl[_epoch, _file]), int(time.time()-__time), int(time.time()-_time)))
                print('OBJ :: {0:.2f},\t LL :: {1:.2f},\t KL :: {2:.2f},\t time :: {3},\t {4}'.format(np.mean(logg_losses[_epoch, _file]), np.mean(logg_ll[_epoch, _file]), np.mean(logg_kl[_epoch, _file]), int(time.time()-__time), int(time.time()-_time)))
                # print('TIME(s) ::', int(time.time()-__time), '+'+str(int(time.time()-_time)));
        EndOfEpoch()
        if _epoch==sim_epochs-1:
            break;


''' loss function '''

@tf.function
def ForwardLosses_v2(correlated_channel_sample, mlps, params, \
    scale_kl=1.0, scale_ll=1.0, dropout_rate_gist=0.0, _complex=False, _complexify_version=1):
    # https://github.com/crazysal/VariationalRNN/blob/master/VariationalRecurrentNeuralNetwork-master/model.py
    [Phi_prior, Phi_z, Phi_dec, Phi_h, f_theta, Phi_enc] = mlps
    [dim_x, dim_z, dim_g, Dropout] = params
    batches, _frames = correlated_channel_sample.shape[:2]
    obj_list = [] 
    kl_list = []
    loglike_list = []
    g_t_1 = g_0 = tf.zeros(shape=[batches,dim_g]);
    for t in range(_frames):
        ''' Step 2.1 Infer Latent Variable using Sample from dataset and Gist '''
        _shape = [-1]; _shape.extend(Phi_h.input.shape[1:])
        _true_h_t = tf.reshape(correlated_channel_sample[:,t],_shape)
        ho_t = Phi_h(_true_h_t)
        z_t_inf_unsplit = Phi_enc(tf.concat([ho_t,g_t_1],axis=-1))
        mean_z_t_inf, logvar_z_t_inf = splitIntoMeanAndVar(z_t_inf_unsplit)
        z_t_inf = reparameterize(mean_z_t_inf, logvar_z_t_inf)
        
        ''' Step 1.1 Prior of Latent variable '''
        z_t_prior_unsplit = Phi_prior(g_t_1)
        mean_z_t_prior, logvar_z_t_prior = splitIntoMeanAndVar(z_t_prior_unsplit)
        # z_t = reparameterize(mean_z_t_prior, logvar_z_t_prior)
        
        ''' Step 1.2 Generate Sample using Latent Variable and dropped out Gist '''
        g_t_1_dropped = Dropout(rate=dropout_rate_gist)(g_t_1)
        zo_t = Phi_z(z_t_inf)
        h_t_unsplit = Phi_dec(tf.concat([zo_t,g_t_1_dropped], axis=-1))
        mean_h_t, logvar_h_t = splitIntoMeanAndVar(h_t_unsplit)
        # h_t = reparameterize(mean_h_t, logvar_h_t)
        if _complex and _complexify_version==2:
            _abs,_angle = tf.split(mean_h_t, num_or_size_splits=2, axis=1)
            mean_h_t = tf.concat([tf.math.softplus(_abs),_angle], axis=1)
        
        ''' Step 1.3 Generate Gist using True Sample and Reparameterised'''
        # ho_t = Phi_h(h_t)
        g_t = f_theta(tf.concat([ho_t,zo_t,g_t_1], axis=-1))
        
        # Objective
        # Sigma_1 = tf.linalg.diag(tf.exp(logvar_z_t_inf))
        # mean_1 = mean_z_t_inf
        # Sigma_2 = tf.linalg.diag(tf.exp(logvar_z_t_prior))
        # mean_2 = mean_z_t_prior
        # expr 1 : tf.math.log(tf.linalg.det(Sigma_2)/tf.linalg.det(Sigma_1)) = tf.math.reduce_sum(logvar_z_t_prior-logvar_z_t_inf, axis=-1)
        # expr 2 : tf.linalg.trace(tf.linalg.inv(Sigma_2)@Sigma_1) = tf.reduce_sum(tf.exp(logvar_z_t_inf-logvar_z_t_prior),axis=-1)
        # expr 3 : tf.reduce_sum(tf.multiply(tf.linalg.matvec(tf.linalg.inv(Sigma_2), (mean_2-mean_1)), (mean_2-mean_1)),axis=-1)) = tf.reduce_sum(tf.multiply(tf.multiply(tf.exp(-logvar_z_t_prior), (mean_z_t_prior-mean_z_t_inf)), (mean_z_t_prior-mean_z_t_inf)), axis=-1)
        # kl = 0.5*( tf.math.log(tf.linalg.det(Sigma_2)/tf.linalg.det(Sigma_1))\
        #    - dim_z\
        #        + tf.linalg.trace(tf.linalg.inv(Sigma_2)@Sigma_1) \
        #            + tf.reduce_sum(tf.multiply(tf.linalg.matvec(tf.linalg.inv(Sigma_2), (mean_2-mean_1)), (mean_2-mean_1)),axis=-1));
        kl = 0.5*( tf.math.reduce_sum(logvar_z_t_prior-logvar_z_t_inf, axis=-1)\
           - dim_z\
               + tf.reduce_sum(tf.exp(logvar_z_t_inf-logvar_z_t_prior),axis=-1) \
                   + tf.reduce_sum(tf.multiply(tf.multiply(tf.exp(-logvar_z_t_prior), (mean_z_t_prior-mean_z_t_inf)), (mean_z_t_prior-mean_z_t_inf)), axis=-1))
        # p_x_given_z = tfp.distributions.MultivariateNormalDiag(loc=vectorizeChannelSample(mean_h_t,_complex=_complex),scale_diag=vectorizeChannelSample(tf.math.exp(logvar_h_t*0.5),_complex=_complex));
        # loglike = p_x_given_z.log_prob(vectorizeChannelSample(correlated_channel_sample[:,t],_complex=_complex))
        _arg1 = vectorizeChannelSample(correlated_channel_sample[:,t],_complex=_complex)
        _arg2 = vectorizeChannelSample(mean_h_t,_complex=_complex)
        _arg3 = vectorizeChannelSample(logvar_h_t,_complex=_complex)
        loglike = logProbNormalUncorrelated(_arg1,_arg2,_arg3)
        loglike = tf.reduce_sum(loglike,axis=-1)
        # assert np.allclose(loglike, tf.reduce_sum(loglike2,axis=-1))
        obj_t = -scale_kl*kl + scale_ll*loglike
        if _complex:
            obj_t = -scale_kl*kl-tf.reduce_sum(tf.square(_arg1-_arg2), axis=(-1))

        kl_list.append(kl)
        loglike_list.append(loglike)
        obj_list.append(obj_t)
        g_t_1 = g_t
    # maximize objective
    objective = tf.reduce_mean(obj_list);
    return objective, [obj_list, loglike_list, kl_list];





