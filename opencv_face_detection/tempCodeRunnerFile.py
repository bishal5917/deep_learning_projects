featureSet,labels = caer.sep_train(train,IMG_SIZE=IMG_SIZE)

# #normalise the feature set -> (0,1)
# featureSet = caer.normalize(featureSet)
# labels = to_categorical(labels,len(characters))