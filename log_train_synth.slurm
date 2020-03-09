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
[[step      400]]     [[train 3.2866s]]     loss: -1.18057253      [[val 0.856s]]     loss: -1.16678822      
[[step      420]]     [[train 3.2873s]]     loss: -1.27507651      [[val 0.856s]]     loss: -1.2551584       
[[step      440]]     [[train 3.2882s]]     loss: -1.4608672       [[val 0.8564s]]     loss: -1.43824098      
[[step      460]]     [[train 3.2886s]]     loss: -1.5542122       [[val 0.8569s]]     loss: -1.53637642      
[[step      480]]     [[train 3.2889s]]     loss: -1.62191501      [[val 0.8566s]]     loss: -1.59742694      
[[step      500]]     [[train 3.2892s]]     loss: -1.59222179      [[val 0.8569s]]     loss: -1.5673033       
[[step      520]]     [[train 3.2883s]]     loss: -1.628041        [[val 0.8566s]]     loss: -1.60734875      
[[step      540]]     [[train 3.2873s]]     loss: -1.67577978      [[val 0.8564s]]     loss: -1.65061323      
[[step      560]]     [[train 3.2876s]]     loss: -1.71702349      [[val 0.8563s]]     loss: -1.68865522      
[[step      580]]     [[train 3.2867s]]     loss: -1.75794621      [[val 0.8565s]]     loss: -1.73008612      
[[step      600]]     [[train 3.2864s]]     loss: -1.87982534      [[val 0.856s]]     loss: -1.85364979      
[[step      620]]     [[train 3.287s]]     loss: -1.93314672      [[val 0.8563s]]     loss: -1.90792488      
[[step      640]]     [[train 3.2869s]]     loss: -1.96418924      [[val 0.8565s]]     loss: -1.94310415      
[[step      660]]     [[train 3.2867s]]     loss: -2.0000878       [[val 0.8564s]]     loss: -1.97958009      
[[step      680]]     [[train 3.2871s]]     loss: -2.02883159      [[val 0.8563s]]     loss: -2.01005131      
[[step      700]]     [[train 3.2873s]]     loss: -2.07205714      [[val 0.8567s]]     loss: -2.05186503      
[[step      720]]     [[train 3.2869s]]     loss: -2.09032395      [[val 0.857s]]     loss: -2.06820139      
[[step      740]]     [[train 3.2875s]]     loss: -2.11615954      [[val 0.8568s]]     loss: -2.09843741      
[[step      760]]     [[train 3.2876s]]     loss: -2.13226206      [[val 0.8566s]]     loss: -2.11621811      
[[step      780]]     [[train 3.2878s]]     loss: -2.17066882      [[val 0.8568s]]     loss: -2.14921271      
[[step      800]]     [[train 3.2872s]]     loss: -2.1832499       [[val 0.8565s]]     loss: -2.16824723      
[[step      820]]     [[train 3.2876s]]     loss: -2.20719535      [[val 0.8563s]]     loss: -2.18651716      
[[step      840]]     [[train 3.2886s]]     loss: -2.23010534      [[val 0.8565s]]     loss: -2.20747066      
[[step      860]]     [[train 3.2885s]]     loss: -2.26529177      [[val 0.8566s]]     loss: -2.24208251      
[[step      880]]     [[train 3.2881s]]     loss: -2.27400917      [[val 0.8567s]]     loss: -2.26302867      
[[step      900]]     [[train 3.2883s]]     loss: -2.29273432      [[val 0.8573s]]     loss: -2.27185888      
[[step      920]]     [[train 3.2882s]]     loss: -2.26622209      [[val 0.8571s]]     loss: -2.25109942      
[[step      940]]     [[train 3.2862s]]     loss: -2.28121253      [[val 0.857s]]     loss: -2.26236287      
[[step      960]]     [[train 3.2855s]]     loss: -2.30090172      [[val 0.857s]]     loss: -2.27610215      
[[step      980]]     [[train 3.2857s]]     loss: -2.31541753      [[val 0.8567s]]     loss: -2.28135541      
[[step     1000]]     [[train 3.2855s]]     loss: -2.35011365      [[val 0.8565s]]     loss: -2.32030735      
[[step     1020]]     [[train 3.2855s]]     loss: -2.42612729      [[val 0.8565s]]     loss: -2.39567366      
[[step     1040]]     [[train 3.2865s]]     loss: -2.44817201      [[val 0.8566s]]     loss: -2.42058442      
[[step     1060]]     [[train 3.2881s]]     loss: -2.44131956      [[val 0.8566s]]     loss: -2.41584872      
[[step     1080]]     [[train 3.2881s]]     loss: -2.45791034      [[val 0.8566s]]     loss: -2.44219895      
[[step     1100]]     [[train 3.2881s]]     loss: -2.41523298      [[val 0.8563s]]     loss: -2.399957        
[[step     1120]]     [[train 3.2878s]]     loss: -2.41218464      [[val 0.8561s]]     loss: -2.3996942       
[[step     1140]]     [[train 3.2876s]]     loss: -2.41332371      [[val 0.8561s]]     loss: -2.40306088      
[[step     1160]]     [[train 3.2868s]]     loss: -2.44745219      [[val 0.8561s]]     loss: -2.43713003      
[[step     1180]]     [[train 3.2874s]]     loss: -2.46093066      [[val 0.856s]]     loss: -2.44161084      
[[step     1200]]     [[train 3.288s]]     loss: -2.52394821      [[val 0.8563s]]     loss: -2.50509541      
[[step     1220]]     [[train 3.2873s]]     loss: -2.55473617      [[val 0.8568s]]     loss: -2.53280155      
[[step     1240]]     [[train 3.2876s]]     loss: -2.57379473      [[val 0.8568s]]     loss: -2.55257553      
[[step     1260]]     [[train 3.2873s]]     loss: -2.57998833      [[val 0.8568s]]     loss: -2.55726832      
[[step     1280]]     [[train 3.2876s]]     loss: -2.59391345      [[val 0.8569s]]     loss: -2.57888257      
[[step     1300]]     [[train 3.2867s]]     loss: -2.59744454      [[val 0.8571s]]     loss: -2.57420897      
[[step     1320]]     [[train 3.2882s]]     loss: -2.61215114      [[val 0.857s]]     loss: -2.58682612      
[[step     1340]]     [[train 3.2884s]]     loss: -2.63877189      [[val 0.8572s]]     loss: -2.61226056      
[[step     1360]]     [[train 3.2881s]]     loss: -2.65259313      [[val 0.8572s]]     loss: -2.63279912      
[[step     1380]]     [[train 3.2847s]]     loss: -2.66813795      [[val 0.857s]]     loss: -2.64863244      
[[step     1400]]     [[train 3.2863s]]     loss: -2.68885882      [[val 0.8568s]]     loss: -2.6766647       
[[step     1420]]     [[train 3.2855s]]     loss: -2.7008363       [[val 0.8568s]]     loss: -2.68840498      
[[step     1440]]     [[train 3.2838s]]     loss: -2.70067928      [[val 0.8568s]]     loss: -2.68887097      
[[step     1460]]     [[train 3.2847s]]     loss: -2.7171509       [[val 0.8567s]]     loss: -2.6979067       
[[step     1480]]     [[train 3.2881s]]     loss: -2.71636083      [[val 0.8566s]]     loss: -2.69536259      
[[step     1500]]     [[train 3.2865s]]     loss: -2.71330271      [[val 0.8566s]]     loss: -2.68315875      
[[step     1520]]     [[train 3.2862s]]     loss: -2.72954062      [[val 0.8565s]]     loss: -2.69376507      
[[step     1540]]     [[train 3.2882s]]     loss: -2.74660102      [[val 0.8564s]]     loss: -2.71451154      
[[step     1560]]     [[train 3.2882s]]     loss: -2.74951568      [[val 0.8564s]]     loss: -2.71978232      
[[step     1580]]     [[train 3.2873s]]     loss: -2.77649217      [[val 0.8567s]]     loss: -2.74558768      
[[step     1600]]     [[train 3.2879s]]     loss: -2.78626205      [[val 0.8567s]]     loss: -2.76109082      
[[step     1620]]     [[train 3.287s]]     loss: -2.79248261      [[val 0.8565s]]     loss: -2.7779659       
[[step     1640]]     [[train 3.286s]]     loss: -2.79777308      [[val 0.8565s]]     loss: -2.7848515       
[[step     1660]]     [[train 3.2846s]]     loss: -2.81468343      [[val 0.8567s]]     loss: -2.80475427      
[[step     1680]]     [[train 3.2849s]]     loss: -2.82621062      [[val 0.8567s]]     loss: -2.81623592      
[[step     1700]]     [[train 3.2849s]]     loss: -2.86107254      [[val 0.8566s]]     loss: -2.85814612      
[[step     1720]]     [[train 3.2858s]]     loss: -2.88281411      [[val 0.8569s]]     loss: -2.8684677       
[[step     1740]]     [[train 3.2866s]]     loss: -2.91224416      [[val 0.8566s]]     loss: -2.89211073      
[[step     1760]]     [[train 3.2874s]]     loss: -2.93162251      [[val 0.8565s]]     loss: -2.90496262      
[[step     1780]]     [[train 3.288s]]     loss: -2.94071895      [[val 0.8569s]]     loss: -2.91461209      
[[step     1800]]     [[train 3.2843s]]     loss: -2.95086053      [[val 0.8569s]]     loss: -2.91550632      
[[step     1820]]     [[train 3.2846s]]     loss: -2.96389289      [[val 0.8569s]]     loss: -2.93064611      
[[step     1840]]     [[train 3.2844s]]     loss: -2.96935997      [[val 0.857s]]     loss: -2.92565649      
[[step     1860]]     [[train 3.2845s]]     loss: -2.96544698      [[val 0.8563s]]     loss: -2.93467341      
[[step     1880]]     [[train 3.2847s]]     loss: -2.98224846      [[val 0.8551s]]     loss: -2.95213112      
[[step     1900]]     [[train 3.2883s]]     loss: -2.98866027      [[val 0.855s]]     loss: -2.96094662      
[[step     1920]]     [[train 3.2889s]]     loss: -2.98683981      [[val 0.8553s]]     loss: -2.96819045      
[[step     1940]]     [[train 3.2891s]]     loss: -2.99174699      [[val 0.8554s]]     loss: -2.98765351      
[[step     1960]]     [[train 3.2891s]]     loss: -3.02296068      [[val 0.8561s]]     loss: -3.0048195       
[[step     1980]]     [[train 3.2889s]]     loss: -3.02625277      [[val 0.8569s]]     loss: -3.00364957      
[[step     2000]]     [[train 3.2895s]]     loss: -3.03776377      [[val 0.8571s]]     loss: -3.02347299      
[[step     2020]]     [[train 3.2893s]]     loss: -3.03330733      [[val 0.8571s]]     loss: -3.01799812      
[[step     2040]]     [[train 3.287s]]     loss: -3.0310046       [[val 0.8572s]]     loss: -3.0098843       
[[step     2060]]     [[train 3.2851s]]     loss: -3.02773123      [[val 0.8571s]]     loss: -3.01180436      
[[step     2080]]     [[train 3.2839s]]     loss: -3.04790535      [[val 0.8569s]]     loss: -3.0343575       
saving model to checkpoints/model
[[step     2100]]     [[train 3.2842s]]     loss: -3.04832991      [[val 0.8568s]]     loss: -3.02507992      
[[step     2120]]     [[train 3.2835s]]     loss: -3.06918664      [[val 0.8568s]]     loss: -3.04379286      
saving model to checkpoints/model
[[step     2140]]     [[train 3.2854s]]     loss: -3.09327713      [[val 0.8563s]]     loss: -3.05968163      
saving model to checkpoints/model
[[step     2160]]     [[train 3.288s]]     loss: -3.11059795      [[val 0.8565s]]     loss: -3.07775697      
saving model to checkpoints/model
[[step     2180]]     [[train 3.2883s]]     loss: -3.09540967      [[val 0.8568s]]     loss: -3.06429363      
[[step     2200]]     [[train 3.2872s]]     loss: -3.10575838      [[val 0.8564s]]     loss: -3.07998636      
saving model to checkpoints/model
[[step     2220]]     [[train 3.288s]]     loss: -3.11727058      [[val 0.8561s]]     loss: -3.08587999      
saving model to checkpoints/model
[[step     2240]]     [[train 3.2857s]]     loss: -3.11663071      [[val 0.8564s]]     loss: -3.09443212      
saving model to checkpoints/model
[[step     2260]]     [[train 3.2837s]]     loss: -3.121223        [[val 0.8564s]]     loss: -3.1018766       
saving model to checkpoints/model
[[step     2280]]     [[train 3.2834s]]     loss: -3.14074575      [[val 0.8558s]]     loss: -3.12022755      
saving model to checkpoints/model
[[step     2300]]     [[train 3.2838s]]     loss: -3.15108585      [[val 0.8563s]]     loss: -3.12774634      
saving model to checkpoints/model
[[step     2320]]     [[train 3.2829s]]     loss: -3.15307648      [[val 0.856s]]     loss: -3.12919441      
saving model to checkpoints/model
[[step     2340]]     [[train 3.2862s]]     loss: -3.16940121      [[val 0.8559s]]     loss: -3.14450242      
saving model to checkpoints/model
[[step     2360]]     [[train 3.2877s]]     loss: -3.17629437      [[val 0.856s]]     loss: -3.14592467      
saving model to checkpoints/model
[[step     2380]]     [[train 3.2885s]]     loss: -3.1740227       [[val 0.8564s]]     loss: -3.14725902      
saving model to checkpoints/model
[[step     2400]]     [[train 3.2876s]]     loss: -3.18777939      [[val 0.8564s]]     loss: -3.16612103      
saving model to checkpoints/model
[[step     2420]]     [[train 3.2879s]]     loss: -3.19659004      [[val 0.8569s]]     loss: -3.17765619      
saving model to checkpoints/model
[[step     2440]]     [[train 3.2873s]]     loss: -3.20848007      [[val 0.857s]]     loss: -3.18653795      
saving model to checkpoints/model
[[step     2460]]     [[train 3.2873s]]     loss: -3.20401536      [[val 0.8568s]]     loss: -3.1859468       
[[step     2480]]     [[train 3.2877s]]     loss: -3.20044573      [[val 0.8569s]]     loss: -3.18480852      
[[step     2500]]     [[train 3.2883s]]     loss: -3.17655683      [[val 0.8569s]]     loss: -3.14984857      
[[step     2520]]     [[train 3.2884s]]     loss: -3.18625083      [[val 0.8568s]]     loss: -3.16416113      
[[step     2540]]     [[train 3.2885s]]     loss: -3.17657254      [[val 0.8566s]]     loss: -3.15579764      
[[step     2560]]     [[train 3.2881s]]     loss: -3.18480551      [[val 0.8568s]]     loss: -3.1636694       
[[step     2580]]     [[train 3.2869s]]     loss: -3.20034916      [[val 0.857s]]     loss: -3.17728243      
[[step     2600]]     [[train 3.2854s]]     loss: -3.22014089      [[val 0.8568s]]     loss: -3.20696693      
saving model to checkpoints/model
[[step     2620]]     [[train 3.2849s]]     loss: -3.23220372      [[val 0.8568s]]     loss: -3.20660109      
[[step     2640]]     [[train 3.2852s]]     loss: -3.24570148      [[val 0.8569s]]     loss: -3.22791681      
saving model to checkpoints/model
[[step     2660]]     [[train 3.2848s]]     loss: -3.26563361      [[val 0.8572s]]     loss: -3.24518511      
saving model to checkpoints/model
[[step     2680]]     [[train 3.2852s]]     loss: -3.2750166       [[val 0.8566s]]     loss: -3.25295543      
saving model to checkpoints/model
[[step     2700]]     [[train 3.2867s]]     loss: -3.28771061      [[val 0.8567s]]     loss: -3.26452741      
saving model to checkpoints/model
[[step     2720]]     [[train 3.2869s]]     loss: -3.27934238      [[val 0.8568s]]     loss: -3.27031717      
saving model to checkpoints/model
[[step     2740]]     [[train 3.2838s]]     loss: -3.2896104       [[val 0.8569s]]     loss: -3.27856052      
saving model to checkpoints/model
[[step     2760]]     [[train 3.2845s]]     loss: -3.27747653      [[val 0.8561s]]     loss: -3.26441938      
[[step     2780]]     [[train 3.2842s]]     loss: -3.2723052       [[val 0.8566s]]     loss: -3.2586652       
[[step     2800]]     [[train 3.2843s]]     loss: -3.26428889      [[val 0.8565s]]     loss: -3.25152891      
[[step     2820]]     [[train 3.284s]]     loss: -3.27765216      [[val 0.8564s]]     loss: -3.25652581      
[[step     2840]]     [[train 3.2869s]]     loss: -3.27577919      [[val 0.8565s]]     loss: -3.24775046      
[[step     2860]]     [[train 3.2867s]]     loss: -3.28792595      [[val 0.8571s]]     loss: -3.2683607       
[[step     2880]]     [[train 3.2878s]]     loss: -3.32560168      [[val 0.8569s]]     loss: -3.29191107      
saving model to checkpoints/model
[[step     2900]]     [[train 3.2879s]]     loss: -3.35297744      [[val 0.8572s]]     loss: -3.31263682      
saving model to checkpoints/model
[[step     2920]]     [[train 3.2882s]]     loss: -3.36777014      [[val 0.8575s]]     loss: -3.3203173       
saving model to checkpoints/model
[[step     2940]]     [[train 3.2879s]]     loss: -3.38404352      [[val 0.8572s]]     loss: -3.33427091      
saving model to checkpoints/model
[[step     2960]]     [[train 3.2872s]]     loss: -3.39543436      [[val 0.8571s]]     loss: -3.34757901      
saving model to checkpoints/model
[[step     2980]]     [[train 3.2864s]]     loss: -3.379458        [[val 0.8564s]]     loss: -3.34649163      
[[step     3000]]     [[train 3.2826s]]     loss: -3.37493677      [[val 0.8563s]]     loss: -3.35357115      
saving model to checkpoints/model
[[step     3020]]     [[train 3.2816s]]     loss: -3.36557782      [[val 0.8557s]]     loss: -3.35519333      
saving model to checkpoints/model
[[step     3040]]     [[train 3.2801s]]     loss: -3.35309879      [[val 0.8558s]]     loss: -3.34767801      
[[step     3060]]     [[train 3.2803s]]     loss: -3.35374282      [[val 0.856s]]     loss: -3.33599559      
[[step     3080]]     [[train 3.2803s]]     loss: -3.3641043       [[val 0.8568s]]     loss: -3.33947039      
[[step     3100]]     [[train 3.2847s]]     loss: -3.36752317      [[val 0.8565s]]     loss: -3.3369818       
[[step     3120]]     [[train 3.2855s]]     loss: -3.37286841      [[val 0.8554s]]     loss: -3.34268126      
[[step     3140]]     [[train 3.2869s]]     loss: -3.38404266      [[val 0.8552s]]     loss: -3.34999745      
[[step     3160]]     [[train 3.2882s]]     loss: -3.38419132      [[val 0.8546s]]     loss: -3.36175943      
saving model to checkpoints/model
[[step     3180]]     [[train 3.2881s]]     loss: -3.38367493      [[val 0.8545s]]     loss: -3.36065697      
[[step     3200]]     [[train 3.2865s]]     loss: -3.39273557      [[val 0.8547s]]     loss: -3.36761895      
saving model to checkpoints/model
[[step     3220]]     [[train 3.2861s]]     loss: -3.39943573      [[val 0.8561s]]     loss: -3.37176418      
saving model to checkpoints/model
[[step     3240]]     [[train 3.2837s]]     loss: -3.40368622      [[val 0.856s]]     loss: -3.38073567      
saving model to checkpoints/model
[[step     3260]]     [[train 3.2833s]]     loss: -3.41338009      [[val 0.8566s]]     loss: -3.38030407      
[[step     3280]]     [[train 3.2833s]]     loss: -3.41957587      [[val 0.8569s]]     loss: -3.38905884      
saving model to checkpoints/model
[[step     3300]]     [[train 3.2828s]]     loss: -3.42129738      [[val 0.8568s]]     loss: -3.38556251      
[[step     3320]]     [[train 3.2799s]]     loss: -3.41634915      [[val 0.8566s]]     loss: -3.38448681      
[[step     3340]]     [[train 3.2832s]]     loss: -3.42235572      [[val 0.8567s]]     loss: -3.39461005      
saving model to checkpoints/model
[[step     3360]]     [[train 3.2829s]]     loss: -3.41174551      [[val 0.8569s]]     loss: -3.37917222      
[[step     3380]]     [[train 3.2825s]]     loss: -3.4100676       [[val 0.8569s]]     loss: -3.37972514      
[[step     3400]]     [[train 3.2836s]]     loss: -3.40986174      [[val 0.8571s]]     loss: -3.39361196      
[[step     3420]]     [[train 3.2874s]]     loss: -3.40412968      [[val 0.8574s]]     loss: -3.38675069      
[[step     3440]]     [[train 3.2867s]]     loss: -3.39364107      [[val 0.8576s]]     loss: -3.37262223      
[[step     3460]]     [[train 3.2859s]]     loss: -3.41459326      [[val 0.8574s]]     loss: -3.39654691      
saving model to checkpoints/model
[[step     3480]]     [[train 3.2867s]]     loss: -3.42271633      [[val 0.8572s]]     loss: -3.40176518      
saving model to checkpoints/model
[[step     3500]]     [[train 3.2874s]]     loss: -3.43635841      [[val 0.8572s]]     loss: -3.40176066      
[[step     3520]]     [[train 3.2864s]]     loss: -3.45480507      [[val 0.857s]]     loss: -3.41428849      
saving model to checkpoints/model
[[step     3540]]     [[train 3.2864s]]     loss: -3.47107218      [[val 0.856s]]     loss: -3.42276874      
saving model to checkpoints/model
[[step     3560]]     [[train 3.2874s]]     loss: -3.46151603      [[val 0.8558s]]     loss: -3.42293404      
saving model to checkpoints/model
[[step     3580]]     [[train 3.287s]]     loss: -3.472585        [[val 0.8561s]]     loss: -3.43532198      
saving model to checkpoints/model
[[step     3600]]     [[train 3.2858s]]     loss: -3.47830068      [[val 0.8559s]]     loss: -3.44366806      
saving model to checkpoints/model
[[step     3620]]     [[train 3.2859s]]     loss: -3.47861428      [[val 0.8557s]]     loss: -3.45288706      
saving model to checkpoints/model
[[step     3640]]     [[train 3.2859s]]     loss: -3.48094711      [[val 0.8567s]]     loss: -3.46204787      
saving model to checkpoints/model
[[step     3660]]     [[train 3.2861s]]     loss: -3.47721304      [[val 0.8568s]]     loss: -3.45624094      
[[step     3680]]     [[train 3.2851s]]     loss: -3.47392266      [[val 0.8567s]]     loss: -3.45182925      
[[step     3700]]     [[train 3.2856s]]     loss: -3.46522717      [[val 0.8566s]]     loss: -3.45057174      
[[step     3720]]     [[train 3.2866s]]     loss: -3.47087911      [[val 0.8568s]]     loss: -3.45427046      
[[step     3740]]     [[train 3.2869s]]     loss: -3.47711347      [[val 0.8565s]]     loss: -3.45581109      
[[step     3760]]     [[train 3.2862s]]     loss: -3.49745335      [[val 0.8565s]]     loss: -3.47548825      
saving model to checkpoints/model
[[step     3780]]     [[train 3.2874s]]     loss: -3.50777804      [[val 0.8567s]]     loss: -3.48421203      
saving model to checkpoints/model
[[step     3800]]     [[train 3.2875s]]     loss: -3.51293273      [[val 0.8566s]]     loss: -3.48617913      
saving model to checkpoints/model
[[step     3820]]     [[train 3.2866s]]     loss: -3.52305188      [[val 0.8566s]]     loss: -3.49206017      
saving model to checkpoints/model
[[step     3840]]     [[train 3.2863s]]     loss: -3.51307564      [[val 0.8566s]]     loss: -3.487673        
[[step     3860]]     [[train 3.2862s]]     loss: -3.51316619      [[val 0.8569s]]     loss: -3.48700596      
[[step     3880]]     [[train 3.2861s]]     loss: -3.51969299      [[val 0.8561s]]     loss: -3.48593562      
[[step     3900]]     [[train 3.2844s]]     loss: -3.52340672      [[val 0.856s]]     loss: -3.48908943      
[[step     3920]]     [[train 3.2848s]]     loss: -3.51930345      [[val 0.8557s]]     loss: -3.48131683      
[[step     3940]]     [[train 3.2848s]]     loss: -3.52035714      [[val 0.8558s]]     loss: -3.48682299      
[[step     3960]]     [[train 3.2862s]]     loss: -3.51984967      [[val 0.8557s]]     loss: -3.48774458      
[[step     3980]]     [[train 3.2857s]]     loss: -3.5209544       [[val 0.8561s]]     loss: -3.50026626      
saving model to checkpoints/model
[[step     4000]]     [[train 3.2865s]]     loss: -3.5273817       [[val 0.8562s]]     loss: -3.50315653      
saving model to checkpoints/model
[[step     4020]]     [[train 3.2858s]]     loss: -3.53563591      [[val 0.8565s]]     loss: -3.52148236      
saving model to checkpoints/model
[[step     4040]]     [[train 3.285s]]     loss: -3.54030484      [[val 0.8564s]]     loss: -3.51750866      
[[step     4060]]     [[train 3.2838s]]     loss: -3.54868507      [[val 0.8565s]]     loss: -3.52295305      
saving model to checkpoints/model
[[step     4080]]     [[train 3.2852s]]     loss: -3.53751557      [[val 0.8564s]]     loss: -3.50994881      
[[step     4100]]     [[train 3.2853s]]     loss: -3.54255559      [[val 0.8564s]]     loss: -3.51168983      
[[step     4120]]     [[train 3.286s]]     loss: -3.54435175      [[val 0.8563s]]     loss: -3.5078333       
[[step     4140]]     [[train 3.2863s]]     loss: -3.55399382      [[val 0.8562s]]     loss: -3.51892161      
[[step     4160]]     [[train 3.2863s]]     loss: -3.54922187      [[val 0.856s]]     loss: -3.52309747      
saving model to checkpoints/model
[[step     4180]]     [[train 3.2854s]]     loss: -3.55238432      [[val 0.8561s]]     loss: -3.53181659      
saving model to checkpoints/model
[[step     4200]]     [[train 3.2857s]]     loss: -3.55863635      [[val 0.8563s]]     loss: -3.53804711      
saving model to checkpoints/model
[[step     4220]]     [[train 3.2852s]]     loss: -3.55273377      [[val 0.8563s]]     loss: -3.53863926      
saving model to checkpoints/model
[[step     4240]]     [[train 3.2856s]]     loss: -3.56797197      [[val 0.8561s]]     loss: -3.5516442       
saving model to checkpoints/model
[[step     4260]]     [[train 3.2859s]]     loss: -3.54727928      [[val 0.8554s]]     loss: -3.51890266      
[[step     4280]]     [[train 3.2854s]]     loss: -3.56134485      [[val 0.8552s]]     loss: -3.52135967      
[[step     4300]]     [[train 3.2857s]]     loss: -3.55115484      [[val 0.8553s]]     loss: -3.52061281      
[[step     4320]]     [[train 3.2864s]]     loss: -3.56376085      [[val 0.8553s]]     loss: -3.52379873      
[[step     4340]]     [[train 3.2863s]]     loss: -3.55874791      [[val 0.8554s]]     loss: -3.52434711      
[[step     4360]]     [[train 3.2861s]]     loss: -3.59663294      [[val 0.8562s]]     loss: -3.56500209      
saving model to checkpoints/model
[[step     4380]]     [[train 3.2858s]]     loss: -3.58551225      [[val 0.8563s]]     loss: -3.5601476       
[[step     4400]]     [[train 3.2861s]]     loss: -3.57964833      [[val 0.856s]]     loss: -3.55476843      
[[step     4420]]     [[train 3.2853s]]     loss: -3.56206373      [[val 0.8558s]]     loss: -3.55315483      
[[step     4440]]     [[train 3.2851s]]     loss: -3.55863931      [[val 0.8562s]]     loss: -3.55119097      
[[step     4460]]     [[train 3.2852s]]     loss: -3.55412311      [[val 0.8556s]]     loss: -3.55090344      
[[step     4480]]     [[train 3.2858s]]     loss: -3.57826922      [[val 0.8557s]]     loss: -3.56506803      
saving model to checkpoints/model
[[step     4500]]     [[train 3.2866s]]     loss: -3.59990818      [[val 0.856s]]     loss: -3.57426275      
saving model to checkpoints/model
[[step     4520]]     [[train 3.2879s]]     loss: -3.623417        [[val 0.8562s]]     loss: -3.58444584      
saving model to checkpoints/model
[[step     4540]]     [[train 3.288s]]     loss: -3.62306819      [[val 0.8559s]]     loss: -3.58436188      
[[step     4560]]     [[train 3.2882s]]     loss: -3.62771519      [[val 0.8566s]]     loss: -3.58300399      
[[step     4580]]     [[train 3.2882s]]     loss: -3.6211694       [[val 0.8567s]]     loss: -3.58300859      
[[step     4600]]     [[train 3.2851s]]     loss: -3.61473491      [[val 0.8565s]]     loss: -3.58636999      
saving model to checkpoints/model
[[step     4620]]     [[train 3.2845s]]     loss: -3.61817931      [[val 0.8561s]]     loss: -3.59205145      
saving model to checkpoints/model
[[step     4640]]     [[train 3.2845s]]     loss: -3.63149664      [[val 0.8562s]]     loss: -3.59761799      
saving model to checkpoints/model
[[step     4660]]     [[train 3.2841s]]     loss: -3.63457448      [[val 0.8561s]]     loss: -3.60726078      
saving model to checkpoints/model
[[step     4680]]     [[train 3.2844s]]     loss: -3.63365044      [[val 0.8561s]]     loss: -3.61032309      
saving model to checkpoints/model
[[step     4700]]     [[train 3.2864s]]     loss: -3.65029504      [[val 0.8562s]]     loss: -3.61551404      
saving model to checkpoints/model
[[step     4720]]     [[train 3.285s]]     loss: -3.647923        [[val 0.8567s]]     loss: -3.61175541      
[[step     4740]]     [[train 3.2844s]]     loss: -3.63000363      [[val 0.8567s]]     loss: -3.59914307      
[[step     4760]]     [[train 3.2851s]]     loss: -3.6158253       [[val 0.8568s]]     loss: -3.57939287      
[[step     4780]]     [[train 3.2844s]]     loss: -3.6058631       [[val 0.8561s]]     loss: -3.57253151      
[[step     4800]]     [[train 3.2855s]]     loss: -3.58762022      [[val 0.856s]]     loss: -3.56453448      
[[step     4820]]     [[train 3.2868s]]     loss: -3.58928811      [[val 0.8561s]]     loss: -3.56260684      
[[step     4840]]     [[train 3.2878s]]     loss: -3.6176361       [[val 0.8562s]]     loss: -3.58048506      
[[step     4860]]     [[train 3.2875s]]     loss: -3.64365994      [[val 0.8561s]]     loss: -3.61032256      
[[step     4880]]     [[train 3.2889s]]     loss: -3.66050283      [[val 0.8571s]]     loss: -3.62430022      
saving model to checkpoints/model
[[step     4900]]     [[train 3.2878s]]     loss: -3.68328446      [[val 0.8568s]]     loss: -3.64119884      
saving model to checkpoints/model
[[step     4920]]     [[train 3.2877s]]     loss: -3.68080732      [[val 0.8569s]]     loss: -3.64570747      
saving model to checkpoints/model
[[step     4940]]     [[train 3.2877s]]     loss: -3.67606366      [[val 0.8568s]]     loss: -3.64369656      
[[step     4960]]     [[train 3.2877s]]     loss: -3.67727787      [[val 0.8569s]]     loss: -3.64199039      
[[step     4980]]     [[train 3.2884s]]     loss: -3.67925349      [[val 0.8563s]]     loss: -3.6412467       
[[step     5000]]     [[train 3.2905s]]     loss: -3.67061445      [[val 0.8569s]]     loss: -3.63905285      
[[step     5020]]     [[train 3.2913s]]     loss: -3.6814611       [[val 0.8565s]]     loss: -3.64111773      
[[step     5040]]     [[train 3.2918s]]     loss: -3.68836611      [[val 0.8563s]]     loss: -3.65301013      
saving model to checkpoints/model
[[step     5060]]     [[train 3.2938s]]     loss: -3.69201999      [[val 0.8562s]]     loss: -3.65821388      
saving model to checkpoints/model
[[step     5080]]     [[train 3.2938s]]     loss: -3.68419531      [[val 0.8567s]]     loss: -3.64758233      
[[step     5100]]     [[train 3.2937s]]     loss: -3.68285519      [[val 0.8566s]]     loss: -3.64901765      
[[step     5120]]     [[train 3.2943s]]     loss: -3.68351566      [[val 0.8567s]]     loss: -3.65826193      
saving model to checkpoints/model
[[step     5140]]     [[train 3.2957s]]     loss: -3.68120736      [[val 0.8568s]]     loss: -3.6527664       
[[step     5160]]     [[train 3.2958s]]     loss: -3.6765328       [[val 0.8559s]]     loss: -3.65492167      
[[step     5180]]     [[train 3.2959s]]     loss: -3.68788075      [[val 0.8557s]]     loss: -3.67637597      
saving model to checkpoints/model
[[step     5200]]     [[train 3.2957s]]     loss: -3.69784784      [[val 0.856s]]     loss: -3.67132265      
[[step     5220]]     [[train 3.2955s]]     loss: -3.68485326      [[val 0.8559s]]     loss: -3.65761329      
[[step     5240]]     [[train 3.2948s]]     loss: -3.68812424      [[val 0.8556s]]     loss: -3.66570107      
[[step     5260]]     [[train 3.2952s]]     loss: -3.68139358      [[val 0.8566s]]     loss: -3.65690489      
[[step     5280]]     [[train 3.2951s]]     loss: -3.68950913      [[val 0.8567s]]     loss: -3.65747022      
[[step     5300]]     [[train 3.2954s]]     loss: -3.68805434      [[val 0.8565s]]     loss: -3.66657935      
[[step     5320]]     [[train 3.2956s]]     loss: -3.7048764       [[val 0.8565s]]     loss: -3.67355023      
[[step     5340]]     [[train 3.2973s]]     loss: -3.70806648      [[val 0.857s]]     loss: -3.67496339      
[[step     5360]]     [[train 3.2961s]]     loss: -3.71055249      [[val 0.8571s]]     loss: -3.67003762      
[[step     5380]]     [[train 3.2956s]]     loss: -3.70626307      [[val 0.857s]]     loss: -3.66950221      
[[step     5400]]     [[train 3.2955s]]     loss: -3.70604544      [[val 0.8569s]]     loss: -3.66839968      
[[step     5420]]     [[train 3.2956s]]     loss: -3.71034737      [[val 0.8568s]]     loss: -3.67873119      
saving model to checkpoints/model
[[step     5440]]     [[train 3.2935s]]     loss: -3.71562596      [[val 0.8568s]]     loss: -3.68171219      
saving model to checkpoints/model
[[step     5460]]     [[train 3.2937s]]     loss: -3.720864        [[val 0.856s]]     loss: -3.68768219      
saving model to checkpoints/model
[[step     5480]]     [[train 3.2948s]]     loss: -3.72723116      [[val 0.8561s]]     loss: -3.68955845      
saving model to checkpoints/model
[[step     5500]]     [[train 3.2945s]]     loss: -3.72795737      [[val 0.8562s]]     loss: -3.69231553      
saving model to checkpoints/model
[[step     5520]]     [[train 3.2901s]]     loss: -3.72258621      [[val 0.8563s]]     loss: -3.6950439       
saving model to checkpoints/model
[[step     5540]]     [[train 3.2896s]]     loss: -3.72296742      [[val 0.8555s]]     loss: -3.69265697      
[[step     5560]]     [[train 3.2902s]]     loss: -3.72728187      [[val 0.8562s]]     loss: -3.69094263      
[[step     5580]]     [[train 3.29s]]     loss: -3.7199358       [[val 0.8558s]]     loss: -3.69219787      
[[step     5600]]     [[train 3.2904s]]     loss: -3.72365885      [[val 0.856s]]     loss: -3.69115688      
[[step     5620]]     [[train 3.2949s]]     loss: -3.72789804      [[val 0.856s]]     loss: -3.68796302      
[[step     5640]]     [[train 3.2963s]]     loss: -3.72470004      [[val 0.8565s]]     loss: -3.68749394      
[[step     5660]]     [[train 3.2965s]]     loss: -3.72269178      [[val 0.8565s]]     loss: -3.69548806      
saving model to checkpoints/model
[[step     5680]]     [[train 3.2955s]]     loss: -3.72503391      [[val 0.8564s]]     loss: -3.6936471       
[[step     5700]]     [[train 3.2955s]]     loss: -3.74117837      [[val 0.8562s]]     loss: -3.70773419      
saving model to checkpoints/model
[[step     5720]]     [[train 3.2954s]]     loss: -3.7235442       [[val 0.8561s]]     loss: -3.69390099      
[[step     5740]]     [[train 3.2947s]]     loss: -3.72395763      [[val 0.8562s]]     loss: -3.69690938      
[[step     5760]]     [[train 3.2944s]]     loss: -3.73107547      [[val 0.8562s]]     loss: -3.7001883       
[[step     5780]]     [[train 3.2947s]]     loss: -3.73197208      [[val 0.8562s]]     loss: -3.70567077      
[[step     5800]]     [[train 3.2933s]]     loss: -3.73078213      [[val 0.8553s]]     loss: -3.70515218      
[[step     5820]]     [[train 3.2935s]]     loss: -3.73288261      [[val 0.8554s]]     loss: -3.71217815      
saving model to checkpoints/model
[[step     5840]]     [[train 3.294s]]     loss: -3.73581539      [[val 0.855s]]     loss: -3.71434755      
saving model to checkpoints/model
[[step     5860]]     [[train 3.2934s]]     loss: -3.73608885      [[val 0.855s]]     loss: -3.70939063      
[[step     5880]]     [[train 3.2941s]]     loss: -3.74384611      [[val 0.8555s]]     loss: -3.7125981       
[[step     5900]]     [[train 3.2958s]]     loss: -3.73786843      [[val 0.8564s]]     loss: -3.71115623      
[[step     5920]]     [[train 3.2959s]]     loss: -3.76398787      [[val 0.8564s]]     loss: -3.72494588      
saving model to checkpoints/model
[[step     5940]]     [[train 3.2942s]]     loss: -3.76821444      [[val 0.8572s]]     loss: -3.72823014      
saving model to checkpoints/model
[[step     5960]]     [[train 3.2946s]]     loss: -3.76625773      [[val 0.8573s]]     loss: -3.7271359       
[[step     5980]]     [[train 3.2942s]]     loss: -3.77688908      [[val 0.857s]]     loss: -3.72981153      
saving model to checkpoints/model
[[step     6000]]     [[train 3.2944s]]     loss: -3.77141375      [[val 0.8569s]]     loss: -3.72754889      
[[step     6020]]     [[train 3.2945s]]     loss: -3.76717565      [[val 0.8572s]]     loss: -3.73066268      
saving model to checkpoints/model
[[step     6040]]     [[train 3.2956s]]     loss: -3.76396842      [[val 0.8566s]]     loss: -3.72135956      
[[step     6060]]     [[train 3.2955s]]     loss: -3.77321899      [[val 0.8564s]]     loss: -3.73989028      
saving model to checkpoints/model
[[step     6080]]     [[train 3.2964s]]     loss: -3.76846479      [[val 0.8565s]]     loss: -3.74196583      
saving model to checkpoints/model
[[step     6100]]     [[train 3.2961s]]     loss: -3.76789782      [[val 0.8565s]]     loss: -3.74392638      
saving model to checkpoints/model
[[step     6120]]     [[train 3.2955s]]     loss: -3.77357826      [[val 0.8557s]]     loss: -3.74431536      
saving model to checkpoints/model
[[step     6140]]     [[train 3.2953s]]     loss: -3.77511261      [[val 0.8561s]]     loss: -3.75044673      
saving model to checkpoints/model
[[step     6160]]     [[train 3.2946s]]     loss: -3.7785009       [[val 0.8561s]]     loss: -3.75318886      
saving model to checkpoints/model
[[step     6180]]     [[train 3.2949s]]     loss: -3.77654049      [[val 0.8558s]]     loss: -3.7527728       
[[step     6200]]     [[train 3.295s]]     loss: -3.7865568       [[val 0.8557s]]     loss: -3.76151461      
saving model to checkpoints/model
[[step     6220]]     [[train 3.2952s]]     loss: -3.78922343      [[val 0.8564s]]     loss: -3.75979619      
[[step     6240]]     [[train 3.2932s]]     loss: -3.79185632      [[val 0.8565s]]     loss: -3.77452614      
saving model to checkpoints/model
[[step     6260]]     [[train 3.2936s]]     loss: -3.79123739      [[val 0.8564s]]     loss: -3.76650014      
[[step     6280]]     [[train 3.2921s]]     loss: -3.79840113      [[val 0.8569s]]     loss: -3.7719416       
[[step     6300]]     [[train 3.2912s]]     loss: -3.81113404      [[val 0.8567s]]     loss: -3.77714108      
saving model to checkpoints/model
[[step     6320]]     [[train 3.2919s]]     loss: -3.81525362      [[val 0.8565s]]     loss: -3.78944785      
saving model to checkpoints/model
[[step     6340]]     [[train 3.2949s]]     loss: -3.81630502      [[val 0.8565s]]     loss: -3.7771666       
[[step     6360]]     [[train 3.2942s]]     loss: -3.82479484      [[val 0.8561s]]     loss: -3.78777338      
[[step     6380]]     [[train 3.2947s]]     loss: -3.82236262      [[val 0.8561s]]     loss: -3.78022579      
[[step     6400]]     [[train 3.2957s]]     loss: -3.81413267      [[val 0.8563s]]     loss: -3.77742643      
[[step     6420]]     [[train 3.2957s]]     loss: -3.80223379      [[val 0.8565s]]     loss: -3.76244735      
[[step     6440]]     [[train 3.2951s]]     loss: -3.81591852      [[val 0.856s]]     loss: -3.77100761      
[[step     6460]]     [[train 3.2964s]]     loss: -3.82099056      [[val 0.8564s]]     loss: -3.76746117      
[[step     6480]]     [[train 3.2961s]]     loss: -3.82133542      [[val 0.8564s]]     loss: -3.77627842      
[[step     6500]]     [[train 3.2954s]]     loss: -3.82812972      [[val 0.8561s]]     loss: -3.78320579      
[[step     6520]]     [[train 3.2942s]]     loss: -3.83757614      [[val 0.8558s]]     loss: -3.78918624      
[[step     6540]]     [[train 3.2946s]]     loss: -3.82770205      [[val 0.8566s]]     loss: -3.79108788      
saving model to checkpoints/model
[[step     6560]]     [[train 3.2936s]]     loss: -3.81970426      [[val 0.8567s]]     loss: -3.79437719      
saving model to checkpoints/model
[[step     6580]]     [[train 3.2941s]]     loss: -3.82340083      [[val 0.8565s]]     loss: -3.79823795      
saving model to checkpoints/model
[[step     6600]]     [[train 3.2945s]]     loss: -3.82612556      [[val 0.8571s]]     loss: -3.79527175      
[[step     6620]]     [[train 3.2956s]]     loss: -3.8313066       [[val 0.8575s]]     loss: -3.80526498      
saving model to checkpoints/model
[[step     6640]]     [[train 3.2952s]]     loss: -3.84574945      [[val 0.8573s]]     loss: -3.81366311      
saving model to checkpoints/model
[[step     6660]]     [[train 3.2954s]]     loss: -3.84458612      [[val 0.857s]]     loss: -3.81409107      
saving model to checkpoints/model
[[step     6680]]     [[train 3.2958s]]     loss: -3.84577556      [[val 0.8572s]]     loss: -3.81481658      
saving model to checkpoints/model
[[step     6700]]     [[train 3.2954s]]     loss: -3.8429759       [[val 0.857s]]     loss: -3.81425193      
[[step     6720]]     [[train 3.2947s]]     loss: -3.84503292      [[val 0.8571s]]     loss: -3.81514703      
saving model to checkpoints/model
[[step     6740]]     [[train 3.2949s]]     loss: -3.82955195      [[val 0.8569s]]     loss: -3.80479829      
[[step     6760]]     [[train 3.2954s]]     loss: -3.84264771      [[val 0.8568s]]     loss: -3.81482519      
[[step     6780]]     [[train 3.2953s]]     loss: -3.84877532      [[val 0.856s]]     loss: -3.81515226      
saving model to checkpoints/model
[[step     6800]]     [[train 3.2954s]]     loss: -3.84352772      [[val 0.8562s]]     loss: -3.80730826      
[[step     6820]]     [[train 3.2961s]]     loss: -3.83391813      [[val 0.8563s]]     loss: -3.79808811      
[[step     6840]]     [[train 3.2968s]]     loss: -3.83717866      [[val 0.8565s]]     loss: -3.79488742      
[[step     6860]]     [[train 3.2969s]]     loss: -3.83030834      [[val 0.8567s]]     loss: -3.78360094      
[[step     6880]]     [[train 3.2962s]]     loss: -3.82907124      [[val 0.8567s]]     loss: -3.78362342      
[[step     6900]]     [[train 3.297s]]     loss: -3.84232838      [[val 0.8568s]]     loss: -3.7960373       
[[step     6920]]     [[train 3.2956s]]     loss: -3.85246825      [[val 0.8567s]]     loss: -3.80713517      
[[step     6940]]     [[train 3.2949s]]     loss: -3.85958598      [[val 0.8563s]]     loss: -3.82110826      
saving model to checkpoints/model
[[step     6960]]     [[train 3.2949s]]     loss: -3.860476        [[val 0.8564s]]     loss: -3.83235528      
saving model to checkpoints/model
[[step     6980]]     [[train 3.2937s]]     loss: -3.86110154      [[val 0.8571s]]     loss: -3.83422673      
saving model to checkpoints/model
[[step     7000]]     [[train 3.2933s]]     loss: -3.86990223      [[val 0.8567s]]     loss: -3.83972302      
saving model to checkpoints/model
[[step     7020]]     [[train 3.2948s]]     loss: -3.87664811      [[val 0.8566s]]     loss: -3.84032812      
saving model to checkpoints/model
[[step     7040]]     [[train 3.2946s]]     loss: -3.88205923      [[val 0.8565s]]     loss: -3.84168526      
saving model to checkpoints/model
[[step     7060]]     [[train 3.2943s]]     loss: -3.88566967      [[val 0.8567s]]     loss: -3.83578846      
[[step     7080]]     [[train 3.2957s]]     loss: -3.88961094      [[val 0.8568s]]     loss: -3.8413099       
[[step     7100]]     [[train 3.295s]]     loss: -3.87978847      [[val 0.857s]]     loss: -3.84192477      
saving model to checkpoints/model
[[step     7120]]     [[train 3.2938s]]     loss: -3.88299055      [[val 0.8569s]]     loss: -3.84858408      
saving model to checkpoints/model
[[step     7140]]     [[train 3.2921s]]     loss: -3.88504351      [[val 0.8572s]]     loss: -3.84919308      
saving model to checkpoints/model
[[step     7160]]     [[train 3.2923s]]     loss: -3.89046558      [[val 0.8569s]]     loss: -3.85834851      
saving model to checkpoints/model
[[step     7180]]     [[train 3.2924s]]     loss: -3.89644441      [[val 0.8568s]]     loss: -3.86708376      
saving model to checkpoints/model
[[step     7200]]     [[train 3.2927s]]     loss: -3.90443275      [[val 0.8568s]]     loss: -3.86479481      
[[step     7220]]     [[train 3.2943s]]     loss: -3.90035945      [[val 0.857s]]     loss: -3.8671652       
saving model to checkpoints/model
[[step     7240]]     [[train 3.2961s]]     loss: -3.89328982      [[val 0.8573s]]     loss: -3.86932735      
saving model to checkpoints/model
[[step     7260]]     [[train 3.2963s]]     loss: -3.8936874       [[val 0.8572s]]     loss: -3.86790472      
[[step     7280]]     [[train 3.2961s]]     loss: -3.88382387      [[val 0.8573s]]     loss: -3.85954543      
[[step     7300]]     [[train 3.2959s]]     loss: -3.88101837      [[val 0.8572s]]     loss: -3.86485325      
[[step     7320]]     [[train 3.2959s]]     loss: -3.87726143      [[val 0.8573s]]     loss: -3.86280247      
[[step     7340]]     [[train 3.2963s]]     loss: -3.89814504      [[val 0.8571s]]     loss: -3.87323701      
saving model to checkpoints/model
[[step     7360]]     [[train 3.2953s]]     loss: -3.89943486      [[val 0.8572s]]     loss: -3.87411499      
saving model to checkpoints/model
[[step     7380]]     [[train 3.2948s]]     loss: -3.91087399      [[val 0.8574s]]     loss: -3.88498484      
saving model to checkpoints/model
[[step     7400]]     [[train 3.2946s]]     loss: -3.9158264       [[val 0.8572s]]     loss: -3.87911185      
[[step     7420]]     [[train 3.2942s]]     loss: -3.91596199      [[val 0.857s]]     loss: -3.87036174      
[[step     7440]]     [[train 3.2935s]]     loss: -3.90189101      [[val 0.8569s]]     loss: -3.8573072       
[[step     7460]]     [[train 3.2944s]]     loss: -3.90619516      [[val 0.8566s]]     loss: -3.86548052      
[[step     7480]]     [[train 3.2949s]]     loss: -3.91071691      [[val 0.8565s]]     loss: -3.86052745      
[[step     7500]]     [[train 3.2948s]]     loss: -3.89539783      [[val 0.8567s]]     loss: -3.85479971      
[[step     7520]]     [[train 3.2949s]]     loss: -3.90088969      [[val 0.8568s]]     loss: -3.85930336      
[[step     7540]]     [[train 3.2957s]]     loss: -3.90840234      [[val 0.8567s]]     loss: -3.8680516       
[[step     7560]]     [[train 3.2948s]]     loss: -3.91109019      [[val 0.8572s]]     loss: -3.86744843      
[[step     7580]]     [[train 3.2954s]]     loss: -3.89208947      [[val 0.857s]]     loss: -3.85155495      
[[step     7600]]     [[train 3.2959s]]     loss: -3.90500097      [[val 0.8563s]]     loss: -3.86602879      
[[step     7620]]     [[train 3.2952s]]     loss: -3.90910633      [[val 0.8563s]]     loss: -3.8680211       
[[step     7640]]     [[train 3.2957s]]     loss: -3.90396356      [[val 0.8564s]]     loss: -3.862071        
[[step     7660]]     [[train 3.2959s]]     loss: -3.88936651      [[val 0.8561s]]     loss: -3.85156012      
[[step     7680]]     [[train 3.2954s]]     loss: -3.91236508      [[val 0.8566s]]     loss: -3.87610516      
[[step     7700]]     [[train 3.2939s]]     loss: -3.9163847       [[val 0.8571s]]     loss: -3.8766083       
[[step     7720]]     [[train 3.2946s]]     loss: -3.92752791      [[val 0.8571s]]     loss: -3.88817249      
saving model to checkpoints/model
[[step     7740]]     [[train 3.294s]]     loss: -3.93801737      [[val 0.8572s]]     loss: -3.89924359      
saving model to checkpoints/model
[[step     7760]]     [[train 3.2945s]]     loss: -3.94813229      [[val 0.8575s]]     loss: -3.90145204      
saving model to checkpoints/model
[[step     7780]]     [[train 3.2942s]]     loss: -3.93923606      [[val 0.8575s]]     loss: -3.89661502      
