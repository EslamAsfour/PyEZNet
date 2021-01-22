import numpy as np
import modules.DataLoader as DataLoader

def GetData():
    print('Loadind data......')
    num_classes = 10
    train_images = DataLoader.train_images() #[60000, 28, 28]
    train_images= np.array(train_images,dtype=np.float32)
    train_labels = DataLoader.train_labels()
    test_images = DataLoader.test_images()
    test_images= np.array(test_images,dtype=np.float32)
    test_labels = DataLoader.test_labels()

    print('Preparing data......')
    '''
    Reshaping and normalization training dataset and its labels
    '''
    training_data = train_images.reshape(60000, 1, 28, 28)
    training_data=training_data/255
    training_labels= train_labels.reshape (-1,1)

    '''
    Reshaping and normalization testing dataset and its labels
    '''
    testing_data = test_images.reshape(10000, 1, 28, 28)
    testing_data=testing_data/255
    testing_labels=test_labels .reshape (-1,1)
    
    return training_data,training_labels,testing_data,testing_labels
