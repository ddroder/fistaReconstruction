from classifier import classifier
i=classifier()
xtrain,xtest,ytrain,ytest=i.train_test(fista=True)
# plt.imshow(xtrain[0],cmap='gray')
# plt.show()
i.trainModel(130,xtrain,xtest,ytrain,ytest,saveModel=True)