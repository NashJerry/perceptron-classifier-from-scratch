To shuffle the training to test data, uncomment ".sample(frac=1)" in like 5 and 6. a seed can also be added using "random_state="

The perceptronTrain function looks like the following:
PerceptronTrain(X, 'y', Z)

X should not be altered.

To run the perceptron, within the call of function PerceptronTrain:
to run a binary test, replace y with 1vs1. 
To make this Class-1 vs Class-2 replace Z with TargetNames.
To make this Class-1 vs Class-3 replace Z with TargetNames2.
To make this Class-2 vs Class-3 replace Z with TargetNames3.

To run a 1vsRest approach:
replace y with multiclass

To add L2 regularisation:
replace y with L2

