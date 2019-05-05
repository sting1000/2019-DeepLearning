from models import *
import torch
import dlc_practical_prologue as helper

# import data
train_input, train_target, train_classes, test_input, test_target, test_classes = helper.generate_pair_sets(1000)

# data augmentation and shuffling
augumented_data = rotated_dataset(train_input).__rotate__()
shuffle_index = torch.randperm(1000)
train_input = torch.cat((train_input,augumented_data.data),0)
train_classes = torch.cat((train_classes, train_classes), 0)
train_target = torch.cat((train_target, train_target), 0)

train_input, train_classes, train_target = train_input[shuffle_index], train_classes[shuffle_index], train_target[shuffle_index]

augumented_data = rotated_dataset(test_input).__rotate__()
shuffle_index = torch.randperm(1000)
test_input = torch.cat((test_input,augumented_data.data),0)
test_classes = torch.cat((test_classes, test_classes), 0)
test_target = torch.cat((test_target, test_target), 0)
test_input, test_classes, test_target = test_input[shuffle_index], test_classes[shuffle_index], test_target[shuffle_index]

# create data dictionary
train_set = {}
train_set['input'] = train_input
train_set['classes'] = train_classes
train_set['target'] = train_target

test_set = {}
test_set['input'] = test_input
test_set['classes'] = test_classes
test_set['target'] = test_target

# running models
print('\nNow running the base line model:\n')
model_base, last_layer_base = ModelTest(baseline(),train_set,test_set,
                                           auxiliary_loss=False,mini_batch_size=100,EPOCH=25,LR=0.001)

print('\nNow running the base line model with auxiliary loss:\n')
model_base_AL, last_layer_base_AL = ModelTest(baseline(),train_set,test_set,
                                           auxiliary_loss=True,mini_batch_size=100,EPOCH=25,LR=0.001)

print('\nNow running the Siamese model without auxiliary loss:\n')
model_siamese_noaux, last_layer_siamese_noaux = ModelTest(CNN_Siamese(),train_set,test_set,
                                           auxiliary_loss=False,mini_batch_size=100,EPOCH=25,LR=0.001)

print('\nNow running the Siamese model with auxiliary loss:\n')
model_siamese, last_layer_siamese = ModelTest(CNN_Siamese(),train_set,test_set,
                                           auxiliary_loss=True,mini_batch_size=100,EPOCH=25,LR=0.001)

