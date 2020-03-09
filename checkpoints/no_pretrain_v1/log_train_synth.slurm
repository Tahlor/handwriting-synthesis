/fslhome/tarch/.conda/envs/tf16/bin/python
/fslhome/tarch/.local/lib/python3.6/site-packages/sklearn/model_selection/_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
  FutureWarning)
train size 9291
val size 489
test size 9780

new run with parameters:
{'attention_mixture_components': 10,
 'batch_size': 32,
 'batch_sizes': [32, 64, 64],
 'beta1_decay': 0.9,
 'beta1_decays': [0.9, 0.9, 0.9],
 'checkpoint_dir': '/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1',
 'early_stopping_steps': 1500,
 'enable_parameter_averaging': False,
 'grad_clip': 10,
 'keep_prob_scalar': 1.0,
 'learning_rate': 0.0002,
 'learning_rates': [0.0002, 5e-05, 2e-05],
 'log_dir': 'logs',
 'log_interval': 20,
 'logging_level': 20,
 'loss_averaging_window': 100,
 'lstm_size': 400,
 'min_steps_to_checkpoint': 2000,
 'num_restarts': 2,
 'num_training_steps': 100000,
 'optimizer': 'rms',
 'output_mixture_components': 20,
 'output_units': 121,
 'patiences': [1500, 1000, 500],
 'prediction_dir': 'predictions',
 'reader': <__main__.DataReader object at 0x7fab4e045ef0>,
 'regularization_constant': 0.0,
 'restart_idx': 0,
 'validation_batch_size': 32,
 'warm_start_init_step': '0'}
all parameters:
[('Variable:0', []),
 ('Variable_1:0', []),
 ('Variable_2:0', []),
 ('rnn/LSTMAttentionCell/lstm_cell/kernel:0', [476, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell/bias:0', [1600]),
 ('rnn/LSTMAttentionCell/attention/weights:0', [476, 30]),
 ('rnn/LSTMAttentionCell/attention/biases:0', [30]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/kernel:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/bias:0', [1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/kernel:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/bias:0', [1600]),
 ('rnn/gmm/weights:0', [400, 121]),
 ('rnn/gmm/biases:0', [121]),
 ('rnn/LSTMAttentionCell/lstm_cell/kernel/RMSProp:0', [476, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell/kernel/RMSProp_1:0', [476, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell/bias/RMSProp:0', [1600]),
 ('rnn/LSTMAttentionCell/lstm_cell/bias/RMSProp_1:0', [1600]),
 ('rnn/LSTMAttentionCell/attention/weights/RMSProp:0', [476, 30]),
 ('rnn/LSTMAttentionCell/attention/weights/RMSProp_1:0', [476, 30]),
 ('rnn/LSTMAttentionCell/attention/biases/RMSProp:0', [30]),
 ('rnn/LSTMAttentionCell/attention/biases/RMSProp_1:0', [30]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/kernel/RMSProp:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/kernel/RMSProp_1:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/bias/RMSProp:0', [1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/bias/RMSProp_1:0', [1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/kernel/RMSProp:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/kernel/RMSProp_1:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/bias/RMSProp:0', [1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/bias/RMSProp_1:0', [1600]),
 ('rnn/gmm/weights/RMSProp:0', [400, 121]),
 ('rnn/gmm/weights/RMSProp_1:0', [400, 121]),
 ('rnn/gmm/biases/RMSProp:0', [121]),
 ('rnn/gmm/biases/RMSProp_1:0', [121])]
trainable parameters:
[('rnn/LSTMAttentionCell/lstm_cell/kernel:0', [476, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell/bias:0', [1600]),
 ('rnn/LSTMAttentionCell/attention/weights:0', [476, 30]),
 ('rnn/LSTMAttentionCell/attention/biases:0', [30]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/kernel:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_1/bias:0', [1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/kernel:0', [876, 1600]),
 ('rnn/LSTMAttentionCell/lstm_cell_2/bias:0', [1600]),
 ('rnn/gmm/weights:0', [400, 121]),
 ('rnn/gmm/biases:0', [121])]
trainable parameter count:
3632431
2020-03-09 17:54:21.023875: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2020-03-09 17:54:21.171274: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties: 
name: Tesla P100-PCIE-16GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:82:00.0
totalMemory: 15.90GiB freeMemory: 15.64GiB
2020-03-09 17:54:21.171364: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2020-03-09 17:54:21.557492: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15165 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:82:00.0, compute capability: 6.0)
built graph
restoring model from /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1/model-0
INFO:tensorflow:Restoring parameters from /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1/model-0
Restoring parameters from /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1/model-0
2020-03-09 17:54:22.003893: W tensorflow/core/framework/op_kernel.cc:1202] OP_REQUIRES failed at save_restore_tensor.cc:170 : Not found: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1/model-0
Traceback (most recent call last):
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1361, in _do_call
    return fn(*args)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1340, in _run_fn
    target_list, status, run_metadata)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py", line 516, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.NotFoundError: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1/model-0
	 [[Node: save/RestoreV2 = RestoreV2[dtypes=[DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save/Const_0_0, save/RestoreV2/tensor_names, save/RestoreV2/shape_and_slices)]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "rnn.py", line 261, in <module>
    nn.fit()
  File "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/tf_base_model.py", line 125, in fit
    self.restore(self.warm_start_init_step)
  File "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/tf_base_model.py", line 332, in restore
    saver.restore(self.session, model_path)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/training/saver.py", line 1755, in restore
    {self.saver_def.filename_tensor_name: save_path})
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 905, in run
    run_metadata_ptr)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1137, in _run
    feed_dict_tensor, options, run_metadata)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1355, in _do_run
    options, run_metadata)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1374, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.NotFoundError: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1/model-0
	 [[Node: save/RestoreV2 = RestoreV2[dtypes=[DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save/Const_0_0, save/RestoreV2/tensor_names, save/RestoreV2/shape_and_slices)]]

Caused by op 'save/RestoreV2', defined at:
  File "rnn.py", line 259, in <module>
    attention_mixture_components=10
  File "rnn.py", line 90, in __init__
    super(rnn, self).__init__(**kwargs)
  File "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/tf_base_model.py", line 108, in __init__
    self.graph = self.build_graph()
  File "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/tf_base_model.py", line 402, in build_graph
    self.saver = tf.train.Saver(max_to_keep=1)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/training/saver.py", line 1293, in __init__
    self.build()
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/training/saver.py", line 1302, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/training/saver.py", line 1339, in _build
    build_save=build_save, build_restore=build_restore)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/training/saver.py", line 796, in _build_internal
    restore_sequentially, reshape)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/training/saver.py", line 449, in _AddRestoreOps
    restore_sequentially)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/training/saver.py", line 847, in bulk_restore
    return io_ops.restore_v2(filename_tensor, names, slices, dtypes)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/ops/gen_io_ops.py", line 1030, in restore_v2
    shape_and_slices=shape_and_slices, dtypes=dtypes, name=name)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 3271, in create_op
    op_def=op_def)
  File "/fslhome/tarch/.conda/envs/tf16/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1650, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

NotFoundError (see above for traceback): Unsuccessful TensorSliceReader constructor: Failed to find any matching files for /panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/no_pretrain_v1/model-0
	 [[Node: save/RestoreV2 = RestoreV2[dtypes=[DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save/Const_0_0, save/RestoreV2/tensor_names, save/RestoreV2/shape_and_slices)]]

Loading from checkpoints/original
0it [00:00, ?it/s]3it [00:00, 23.51it/s]4it [00:00, 27.82it/s]
0it [00:00, ?it/s]9it [00:00, 89.74it/s]21it [00:00, 96.97it/s]32it [00:00, 99.54it/s]44it [00:00, 103.50it/s]48it [00:00, 109.46it/s]
