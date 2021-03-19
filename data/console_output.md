```
root@code-machine:/dockerx# python main.py --config config/config.yaml
2021-03-19 08:17:31,943 Getting config parameters...
2021-03-19 08:17:31,943 -----------config parameters------------
2021-03-19 08:17:31,943 ---split_ratio = 0.2
2021-03-19 08:17:31,943 ---random_state = 0
2021-03-19 08:17:31,943 ---max_len = 128
2021-03-19 08:17:31,943 Loading training and test data...
2021-03-19 08:17:32,949 Parsing input text...
2021-03-19 08:17:42,473 Spliting datasets, the split ratio is 0.2, random state is 0
2021-03-19 08:17:42,488 Parsing test dataset...
2021-03-19 08:17:50,943 Loading done.
2021-03-19 08:17:50,952 Vecterizing data for neural network training...
2021-03-19 08:17:50,952 Detected pretrained embedding.
2021-03-19 08:17:50,952 Loading pretrained embeddings ./data/glove.twitter.27B.100d.txt
2021-03-19 08:18:09,607 Loading done
2021-03-19 08:18:09,608 Creating vocabulary...
2021-03-19 08:18:10,654 Done. Got 1193516 words
2021-03-19 08:18:10,654 Preparing data for training...
2021-03-19 08:18:17,576 Creating textCNN model...
2021-03-19 08:18:17,577 -------Training config parameters-------
2021-03-19 08:18:17,577 ---model_name = text_cnn
2021-03-19 08:18:17,577 ---max_input_len = 128
2021-03-19 08:18:17,577 ---batch_size = 32
2021-03-19 08:18:17,577 ---dropout = 0.5
2021-03-19 08:18:17,577 ---epochs = 2
2021-03-19 08:18:17,577 ---num_of_classes = 6
2021-03-19 08:18:17.588146: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library libamdhip64.so
2021-03-19 08:18:17.610269: E tensorflow/stream_executor/rocm/rocm_driver.cc:982] could not retrieve ROCM device count: HIP_ERROR_NoDevice
2021-03-19 08:18:17.610287: E tensorflow/stream_executor/rocm/rocm_driver.cc:982] could not retrieve ROCM device count: HIP_ERROR_NoDevice
2021-03-19 08:18:17.625496: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU Frequency: 3600000000 Hz
2021-03-19 08:18:17.625726: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x53e4340 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-03-19 08:18:17.625739: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-03-19 08:18:17.628062: E tensorflow/stream_executor/rocm/rocm_driver.cc:982] could not retrieve ROCM device count: HIP_ERROR_NoDevice
2021-03-19 08:18:17,642 Found embedding matrix, setting trainable=False
2021-03-19 08:18:17.649745: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 477406400 exceeds 10% of free system memory.
2021-03-19 08:18:17.757340: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 477406400 exceeds 10% of free system memory.
2021-03-19 08:18:17.810677: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 477406400 exceeds 10% of free system memory.
2021-03-19 08:18:18.115397: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 477406400 exceeds 10% of free system memory.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 128, 100)          119351600
_________________________________________________________________
conv1d (Conv1D)              (None, 128, 128)          89728
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 64, 128)           0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 64, 256)           164096
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 32, 256)           0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 32, 512)           393728
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 16, 512)           0
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0
_________________________________________________________________
dense (Dense)                (None, 128)               1048704
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 774
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 42
=================================================================
Total params: 121,048,672
Trainable params: 1,697,072
Non-trainable params: 119,351,600
_________________________________________________________________
2021-03-19 08:18:18.352450: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 65359872 exceeds 10% of free system memory.
2021-03-19 08:18:18.438433: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:18:18.453992: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:18:18.459581: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
Epoch 1/2
2021-03-19 08:18:18.797000: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
3988/3990 [============================>.] - ETA: 0s - accuracy: 0.9429 - loss: 0.07222021-03-19 08:19:32.537729: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:19:32.540700: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:19:32.542893: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
3990/3990 [==============================] - 78s 20ms/step - accuracy: 0.9429 - loss: 0.0721 - val_accuracy: 0.9936 - val_loss: 0.0574
Epoch 2/2
3988/3990 [============================>.] - ETA: 0s - accuracy: 0.9827 - loss: 0.05602021-03-19 08:20:50.951265: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:20:50.954316: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
3990/3990 [==============================] - 78s 20ms/step - accuracy: 0.9827 - loss: 0.0560 - val_accuracy: 0.9936 - val_loss: 0.0544
2021-03-19 08:20:55.740597: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:20:55.743576: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:20:55.745359: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:20:55.810423: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:21:00,344 Accuracy : 0.917248942503525
2021-03-19 08:21:00,344
              precision    recall  f1-score   support

           0       0.90      0.66      0.76      3101
           1       0.62      0.09      0.15       329
           2       0.82      0.70      0.75      1698
           3       1.00      0.00      0.00        91
           4       0.77      0.62      0.68      1594
           5       1.00      0.00      0.00       298

   micro avg       0.84      0.60      0.70      7111
   macro avg       0.85      0.34      0.39      7111
weighted avg       0.84      0.60      0.67      7111
 samples avg       0.99      0.95      0.94      7111


2021-03-19 08:21:00.366439: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:21:00.370282: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:21:00.372061: I tensorflow/core/common_runtime/gpu_fusion_pass.cc:508] ROCm Fusion is enabled.
2021-03-19 08:21:21,554 Saving prediction result to a csv file...
2021-03-19 08:21:22,365 Done. Prediction completed!

```
