import os
import os.path
import train_model


model_name = 'model.pth'
if(os.path.isfile('./' + model_name)):
    print('Model already exists')
    model = train_model.load_model(model_name)
else:
    print('Training model')
    train_model.train_and_save(model_name)

print(model)