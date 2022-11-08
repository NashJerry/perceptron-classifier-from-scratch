import numpy as np
import pandas as pd 

#Training and testing data are imported imported can be randomly shuffled by uncommenting sample(frac=1). The data is converted to a numpy
train = pd.read_csv('train.data', header=None)#.sample(frac=1)#, random_state=3)
test = pd.read_csv('test.data', header = None)#.sample(frac=1)#, random_state=3)
train = train.to_numpy()
test = test.to_numpy()

TargetNames = ['class-1', 'class-2', 'class-3']
TargetNames2 = ['class-1','class-3', 'class-2']
TargetNames3 = ['class-2', 'class-3', 'class-1']

def PerceptronTest(Testb, testW, TestData,pair, testtype, multiData,lvalue,trainData):
    '''
    Runs the binary/multiclass/L2 perceptron.

    Parametres:
        Testb(int) : bias score to be used in the algorithm
        testW (list) : the weight each feature will have
        TestData (ndarray) : the test data
        pair (list) : class names
        testtype (str) : determines which of the three test types will be used
        multiData (lst) : returns a list of lists with weights and biases for L2 and multiclass tests

    Returns:
        test accuracy score
    '''
    inputs1 = TestData[:, :4]            
    ClassName = TestData[:, 4]
    inputsTrain = trainData[:, :4]            
    ClassNameTrain = trainData[:, 4]


    if testtype == '1vs1' : #drops whichever class is in index 2 allowing a 1v1 comparison of the remaining 2 classes

            ClassName[ClassName== pair[0]] = 1 #our target class is +1
            ClassName[ClassName == pair[1]] = -1 #the alternative class is -1
            ClassName[ClassName == pair[2]] = 3 
            TestData =np.delete(TestData, np.where(ClassName == 3), axis = 0)  #the third class is dropped
            inputs1 = TestData[:, :4]
            ClassName = TestData[:, 4]
            assigned = list(zip(inputs1, ClassName)) #creates a new formate with all rows with the class in index 2 removed
            Accuracy = 0 #accuracy initialised as 0


            ClassNameTrain[ClassNameTrain== pair[0]] = 1 #our target class is +1
            ClassNameTrain[ClassNameTrain == pair[1]] = -1 #the alternative class is -1
            ClassNameTrain[ClassNameTrain == pair[2]] = 3 
            trainData =np.delete(trainData, np.where(ClassNameTrain == 3), axis = 0)  #the third class is dropped
            inputsTrain = trainData[:, :4]
            ClassNameTrain = trainData[:, 4]
            assignedTrain = list(zip(inputsTrain, ClassNameTrain)) #creates a new formate with all rows with the class in index 2 removed
            AccuracyTrain= 0 #accuracy initialised as 0

            for row in assigned:
                a =np.dot(testW,row[0]) + Testb
                if row[1]*a > 0: #a positive value is a correct predicted i.2. "-1 * -1 = +1" or "+1 * +1 = +1"
                    Accuracy += 1 #accuract counter is increased by 1
            
            Percentage = (Accuracy/len(assigned))*100 #accuracy is the  number of correct predictions over total predictions
            print("The testing accuracy is in 1vs1 : ", Percentage, "%")

            for row in assignedTrain:
                a =np.dot(testW,row[0]) + Testb
                if row[1]*a > 0: #a positive value is a correct predicted i.2. "-1 * -1 = +1" or "+1 * +1 = +1"
                    AccuracyTrain += 1 #accuract counter is increased by 1
            
            Percentage = (AccuracyTrain/len(assignedTrain))*100 #accuracy is the  number of correct predictions over total predictions
            print("The training accuracy is in 1vs1 : ", Percentage, "%")

    #if the test is not binary this runs
    if testtype == 'multiclass' or testtype == 'L2' :
        inputs = TestData[:, :4].copy() #the first four features are turned inpyuts
        ClassName = TestData[:, 4].copy() #prediction column is renamed
        inputsTrain = trainData[:, :4].copy()
        ClassNameTrain = trainData[:, 4].copy()
        predicclass =[] #the predicted class of each will be stored here
        predicclassTrain = []

        for i in range(0,len(inputs)):
            allActivations = [] #all activation scores will be stored here

            for wb in multiData: #if the weights and biased are in one of the three different combinations
                a =np.dot(wb[1],inputs[i]) + wb[0]
                allActivations.append(a) #the activation score is appended to the list
                argmaxPos = np.argmax(allActivations) #takes the index position of the maximum position
                prePredClass = argmaxPos + 1 #the numerical value of the predicted class is index position plus one due to index 0
                holder = "class-"
                newHolder = holder + str(prePredClass) #the predicted class is now a string
            predicclass.append(newHolder)
            
        ClassNameList = list(ClassName)
        counter = 0

        for i in range(len(ClassNameList)):
            if ClassNameList[i] == predicclass[i]:
                counter +=1 # if the values match in the predicted and the actual data, 1 is added to the accuracy
        MultiClassAccuracy = counter/len(ClassNameList)*100
        if testtype == 'multiclass':
            print("The multiclass accuracy on the Test Data is: ", round(MultiClassAccuracy,2), "%")
        elif testtype =='L2':
            print("The accuracy on the Test Data for L2 =",lvalue,"is",round(MultiClassAccuracy,2), "%")

        for i in range(0,len(inputsTrain)):
            allActivationsTrain = [] #all activation scores will be stored here

            for wb in multiData: #if the weights and biased are in one of the three different combinations
                a =np.dot(wb[1],inputsTrain[i]) + wb[0]
                allActivationsTrain.append(a) #the activation score is appended to the list
                argmaxPosTrain = np.argmax(allActivationsTrain) #takes the index position of the maximum position
                prePredClassTrain = argmaxPosTrain + 1 #the numerical value of the predicted class is index position plus one due to index 0
                holderTrain = "class-"
                newHolderTrain = holderTrain + str(prePredClassTrain) #the predicted class is now a string
            predicclassTrain.append(newHolderTrain)
            
        ClassNameListTrain = list(ClassNameTrain)
        counterTrain = 0

        for i in range(len(ClassNameListTrain)):
            if ClassNameListTrain[i] == predicclassTrain[i]:
                counterTrain +=1 # if the values match in the predicted and the actual data, 1 is added to the accuracy
        MultiClassAccuracyTrain = counterTrain/len(ClassNameListTrain)*100
        if testtype == 'multiclass':
            print("The multiclass accuracy on the Training Data is: ", round(MultiClassAccuracyTrain,2), "%")
        elif testtype =='L2':
            print("The accuracy on the training Data for L2 =",lvalue,"is",round(MultiClassAccuracyTrain,2), "%")


def PerceptronTrain(TrainingData, testType, pair):
    '''
    Trains the perceptron model 20 times. It then returns the bias and weights to be used in testing

    Parametres:
        TrainingData (np.ndarray) : the data that would be used to train the perceptron
        20 (int) :the maximum number of times the algorithm will be run the bias and wights will be adjusted  
        testType (str) : determines whether the algorithm will be binary, multiclass or use an L2
        pair (list) : the list off target names that will be used to train classification ---------

    Returns:
        b (int) = bias that will be used to adjust the overall score of the weights
        W (int) = weight that each feature will be assigned in classification
    
    '''
    #the intial values of the features in our data, stores as an array
    inputs1 = TrainingData[:, :4]
    #assigns the class variables to ClassName
    ClassName = TrainingData[:, 4]
    
    #if the test type is 1v1, the first element of pair will be +1, the second will be -1 while the third (3) will be discarded. the training is updated to reflect this
    if testType == '1vs1':
        L2 =1
        store =0
        ClassName[ClassName== pair[0]] = 1
        ClassName[ClassName == pair[1]] = -1
        ClassName[ClassName == pair[2]] = 3
        TrainingData =np.delete(TrainingData, np.where(ClassName == 3), axis = 0) 
        #a new tables without the dropped column is made
        inputs1 = TrainingData[:, :4]
        ClassName = TrainingData[:, 4]
        assigned = list(zip(inputs1, ClassName))
        W = np.array([0.0,0.0,0.0,0.0])
        b= 0
        for i in range(20):
            for row in assigned:
                a =np.dot(W,row[0]) +b

                #if the activation score is below zero, there has been a misclassification so this updates the weights and the bias
                if row[1]*a <= 0:
                    for wi in range(len(W)):
                        W[wi] = W[wi] +  (row[1]*row[0][wi])
                    b = b + row[1]
        PerceptronTest(b, W,test,pair, testType, store,L2,train) #the test function is then called
     
    if testType == 'multiclass':
        multiclassCombinations = [([TargetNames[i]],[j for j in TargetNames if j != TargetNames[i]]) for i in range(len(TargetNames))] #creates a list with the different combinations of classes
        store = []
        for possibility in multiclassCombinations:
            inputs = TrainingData[:, :4].copy()
            ClassName = TrainingData[:, 4].copy()

            ClassName[ClassName== possibility[0]] = 1
            ClassName[ClassName == possibility[1][0]] = -1
            ClassName[ClassName == possibility[1][1]] = -1
            assigned = list(zip(inputs, ClassName)) 
            W = np.array([0.0,0.0,0.0,0.0])
            b= 0
            
            if testType == 'multiclass':
                L2 = 1
                for i in range(20):
                    for row in assigned:
                        a =np.dot(W,row[0]) + b

                        if row[1]*a <= 0:
                            for wi in range(len(W)):
                                W[wi] = W[wi] +  (row[1]*row[0][wi])
                            b = b + row[1]
                store.append([b,W])
        PerceptronTest(b, W,test,pair, testType, store,L2,train)                

    if testType == 'L2':
        multiclassCombinations = [([TargetNames[i]],[j for j in TargetNames if j != TargetNames[i]]) for i in range(len(TargetNames))]
   
        for L2 in [0.01, 0.1, 1, 10, 100]: #these the 5 potential regulirasation scores which are in an interable 
            store=[] #stores the biases and weights for the particlar combination
            for possibility in multiclassCombinations:
                inputs = TrainingData[:, :4].copy()
                ClassName = TrainingData[:, 4].copy()
                ClassName[ClassName== possibility[0]] = 1
                ClassName[ClassName == possibility[1][0]] = -1
                ClassName[ClassName == possibility[1][1]] = -1
                assigned = list(zip(inputs, ClassName)) 
                W = np.array([0.0,0.0,0.0,0.0])
                b= 0
                for i in range(20):
                    for row in assigned:
                        a =np.dot(W,row[0]) + b
                        if row[1]*a <= 0:
                            for wi in range(len(W)):
                                W[wi] = (1-2*L2)*W[wi]+(row[1]*row[0][wi]) #the L2 is used as a lamba to regularize the lambda
                            b = b + row[1]
                store.append([b,W])   
            PerceptronTest(b, W,test,pair, testType, store,L2,train)    
PerceptronTrain(train,'multiclass',TargetNames)

