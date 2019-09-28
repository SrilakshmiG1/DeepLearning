#AND GATE Implementation using Keras

#Importing the libraries
from keras.layers import Dense,Activation
from keras.models import Sequential,load_model
import numpy as np

#Input data
data=np.array([[0,0],[0,1],[1,0],[1,1],[0,1],[1,1]])
#Output data
label=np.array([[0],[0],[0],[1],[0],[1]])

#Building the model using sequential api method
model=Sequential()
model.add(Dense(1,input_dim=2))
model.add(Activation('relu'))
model.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data,label,epochs=100,batch_size=1)

#To see the score of the model
score=model.evaluate(data,label,batch_size=1)
print(score)

#To see the metric name
print(model.metrics_names)

#To predict the label for new observation
result=model.predict(np.array([[1,0]]))
print(result.astype(int))
      
#To see the weights
weights=model.get_weights()
print(weights)

      
#save the model weights
my_weights=model.save_weights('my_model_weights.h5')
print(my_weights)

#save the model
model.save('my_model.h5') #contains (architecture + weights + optimizer state)

#Loading the saved model
my_model=load_model('my_model.h5')
print(my_model.summary())      
