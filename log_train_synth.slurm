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
 'checkpoint_dir': 'checkpoints',
 'early_stopping_steps': 1500,
 'enable_parameter_averaging': False,
 'grad_clip': 10,
 'keep_prob_scalar': 1.0,
 'learning_rate': 0.0001,
 'learning_rates': [0.0001, 5e-05, 2e-05],
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
 'reader': <__main__.DataReader object at 0x7f4195defac8>,
 'regularization_constant': 0.0,
 'restart_idx': 0,
 'validation_batch_size': 32,
 'warm_start_init_step': 0}
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
2020-03-07 11:57:54.912643: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2020-03-07 11:57:55.066593: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties: 
name: Tesla P100-PCIE-16GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:03:00.0
totalMemory: 15.90GiB freeMemory: 15.64GiB
2020-03-07 11:57:55.066694: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2020-03-07 11:57:56.344174: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15165 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:03:00.0, compute capability: 6.0)
built graph
[[step        0]]     [[train 3.7146s]]     loss: 3.80082273       [[val 1.4016s]]     loss: 3.75667071       
[[step       20]]     [[train 3.262s]]     loss: 3.74429658       [[val 0.8776s]]     loss: 3.75235958       
[[step       40]]     [[train 3.2487s]]     loss: 3.60700816       [[val 0.8659s]]     loss: 3.60392786       
[[step       60]]     [[train 3.2548s]]     loss: 3.45659764       [[val 0.8618s]]     loss: 3.44583644       
[[step       80]]     [[train 3.262s]]     loss: 3.31832404       [[val 0.8606s]]     loss: 3.30875112       
[[step      100]]     [[train 3.2624s]]     loss: 3.20437756       [[val 0.8544s]]     loss: 3.19518939       
[[step      120]]     [[train 3.2722s]]     loss: 2.9969791        [[val 0.8556s]]     loss: 2.98316821       
[[step      140]]     [[train 3.2835s]]     loss: 2.80222025       [[val 0.8562s]]     loss: 2.79405747       
[[step      160]]     [[train 3.2876s]]     loss: 2.56670645       [[val 0.8571s]]     loss: 2.56540322       
[[step      180]]     [[train 3.2889s]]     loss: 2.2299311        [[val 0.8572s]]     loss: 2.23115653       
[[step      200]]     [[train 3.2887s]]     loss: 1.7861744        [[val 0.8569s]]     loss: 1.79025258       
[[step      220]]     [[train 3.2882s]]     loss: 1.2626479        [[val 0.8567s]]     loss: 1.27349648       
[[step      240]]     [[train 3.2878s]]     loss: 0.74504713       [[val 0.8568s]]     loss: 0.75351333       
[[step      260]]     [[train 3.2882s]]     loss: 0.28576992       [[val 0.8565s]]     loss: 0.29923953       
[[step      280]]     [[train 3.2866s]]     loss: -0.1056482       [[val 0.8563s]]     loss: -0.08878601      
[[step      300]]     [[train 3.2868s]]     loss: -0.40795812      [[val 0.8568s]]     loss: -0.39319471      
[[step      320]]     [[train 3.2869s]]     loss: -0.6430306       [[val 0.8569s]]     loss: -0.63438676      
[[step      340]]     [[train 3.2859s]]     loss: -0.76759166      [[val 0.8565s]]     loss: -0.75611047      
[[step      360]]     [[train 3.2851s]]     loss: -0.94516141      [[val 0.8562s]]     loss: -0.93428087      
[[step      380]]     [[train 3.2861s]]     loss: -1.07609588      [[val 0.8564s]]     loss: -1.06738695      
