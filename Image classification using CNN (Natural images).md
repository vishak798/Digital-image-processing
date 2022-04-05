```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
```


```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```


```python
plt.figure(figsize=(3,3))
plt.imshow(X_train[0])

```




    <matplotlib.image.AxesImage at 0x1f82f207400>




    
![png](output_2_1.png)
    



```python
# reshaping
y_train = y_train.reshape(-1,)
```


```python
X_train = X_train/255 
X_test = X_test/255
```

## CNN


```python
cnn = models.Sequential([
    #cnn
    layers.Conv2D(filters=32, kernel_size = (3,3), activation = "relu", input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(filters=64, kernel_size = (3,3), activation = "relu", input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    #dense
    layers.Flatten(),
    layers.Dense(64, activation = "relu"),
    layers.Dense(10, activation = "softmax")
])
```


```python
cnn.compile(optimizer = "adam",
             loss = "sparse_categorical_crossentropy",
             metrics = ["accuracy"])
```


```python
cnn.fit(X_train, y_train, epochs=20)
```

    Epoch 1/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.6247 - accuracy: 0.7839
    Epoch 2/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.6040 - accuracy: 0.7887
    Epoch 3/20
    1563/1563 [==============================] - 14s 9ms/step - loss: 0.5782 - accuracy: 0.7972
    Epoch 4/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.5524 - accuracy: 0.8055
    Epoch 5/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.5319 - accuracy: 0.8146
    Epoch 6/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.5105 - accuracy: 0.8199
    Epoch 7/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.4903 - accuracy: 0.8272
    Epoch 8/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.4691 - accuracy: 0.8342
    Epoch 9/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.4505 - accuracy: 0.8393
    Epoch 10/20
    1563/1563 [==============================] - 17s 11ms/step - loss: 0.4312 - accuracy: 0.8482
    Epoch 11/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.4165 - accuracy: 0.8508
    Epoch 12/20
    1563/1563 [==============================] - 16s 11ms/step - loss: 0.4003 - accuracy: 0.8574
    Epoch 13/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.3866 - accuracy: 0.8618
    Epoch 14/20
    1563/1563 [==============================] - 15s 10ms/step - loss: 0.3742 - accuracy: 0.8659
    Epoch 15/20
    1563/1563 [==============================] - 15s 10ms/step - loss: 0.3593 - accuracy: 0.8707
    Epoch 16/20
    1563/1563 [==============================] - 16s 10ms/step - loss: 0.3467 - accuracy: 0.8755
    Epoch 17/20
    1563/1563 [==============================] - 15s 10ms/step - loss: 0.3331 - accuracy: 0.8813
    Epoch 18/20
    1563/1563 [==============================] - 15s 10ms/step - loss: 0.3223 - accuracy: 0.8836
    Epoch 19/20
    1563/1563 [==============================] - 15s 10ms/step - loss: 0.3105 - accuracy: 0.8883
    Epoch 20/20
    1563/1563 [==============================] - 15s 10ms/step - loss: 0.2994 - accuracy: 0.8926
    




    <keras.callbacks.History at 0x1f82ed9efd0>




```python
cnn.evaluate(X_test, y_test)
```

    313/313 [==============================] - 1s 3ms/step - loss: 1.4418 - accuracy: 0.6792
    




    [1.4417712688446045, 0.6791999936103821]




```python
y_pred = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_pred]
from sklearn.metrics import classification_report
print("Classification Report : \n", classification_report(y_test, y_classes))
```

    Classification Report : 
                   precision    recall  f1-score   support
    
               0       0.72      0.72      0.72      1000
               1       0.83      0.76      0.80      1000
               2       0.59      0.52      0.55      1000
               3       0.47      0.56      0.51      1000
               4       0.60      0.64      0.62      1000
               5       0.59      0.60      0.59      1000
               6       0.77      0.72      0.74      1000
               7       0.77      0.68      0.72      1000
               8       0.79      0.79      0.79      1000
               9       0.73      0.80      0.76      1000
    
        accuracy                           0.68     10000
       macro avg       0.69      0.68      0.68     10000
    weighted avg       0.69      0.68      0.68     10000
    
    


```python

```
