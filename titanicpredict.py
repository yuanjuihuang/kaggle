import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class TitnaicPredictClass:

    #_CONVERT_CIBIN_TABLE = dict()
    PositiveTrainDataSet = None
    NegativeTrainDataSet = None
    TRAIN_LABEL_SET = None
    TRAIN_DATA_SET = None
    TEST_LABEL_SET = None
    TEST_DATA_SET = None
    TrainDataRatio = 0.5
    AGE_NORMALIZATION = 0
    FARE_NORMALIZATION = 0
    SIBSP_NORMALIZATION = 0
    PARCH_NORMALIZATION = 0
    convertTable = dict()

    def __init__(self):
        pass

    def _InitalConvertCabinEncoder(self,convertStr):

        convertStr = re.sub(r'\d', "",convertStr) # remove numeric

        convertStr = re.sub(" ", "", convertStr)

        if str(convertStr) in convertTable:
            return convertTable.get(str(convertStr), 0)
        else:
            convertTable[str(convertStr)] = len(convertTable) + 1
            return convertTable.get(str(convertStr), 0)

    def _NormalizationFeature(self, dataset):
        # Not use this feature
        dataset = dataset.drop(['PassengerId', 'Name', 'Ticket'], axis=1)  # drop axis = 1 (column)

        # Change Pclass numeric from 0 to 1
        dataset['Pclass'] = (dataset['Pclass'] - 1) * 0.5

        # Chagne SibSp numeric from 0 to 1
        dataset['SibSp'] = dataset['SibSp'] / self.SIBSP_NORMALIZATION

        # Change Parch numeric from 0 to 1
        dataset['Parch'] = dataset['Parch'] / self.PARCH_NORMALIZATION

        # chant Embarked from alpha to 0, 1 for differ char , use Onehot encoding
        dataset = pd.get_dummies(data=dataset, columns=["Embarked"])

        # Change Sex feature to 0 or 1
        dataset.Sex = [x.lower() for x in dataset.Sex]  # map : function work for list or DataFrame
        #dataset['Sex'] = (dataset['Sex'] == 'male').astype(int)
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # create new column Miss Fare field ,
        dataset['MissFare'] = (dataset['Fare'].isnull()).astype(int)

        # because Fare column have null value so we need to put zero value
        dataset['Fare'] = dataset['Fare'].fillna(0)  # average 30
        dataset['Fare'] = dataset['Fare'] / self.FARE_NORMALIZATION  # average

        # create new column Miss Age field ,
        dataset['MissAge'] = (dataset['Age'].isnull()).astype(int)
        dataset['Age'] = dataset['Age'].fillna(0)  # average 30
        dataset['Age'] = dataset['Age'] / self.AGE_NORMALIZATION  # average

        # drop Cabin column
        dataset = dataset.drop(['Cabin'], axis=1)
        """
        dataset['Cabin'] = dataset['Cabin'].fillna('None')  # fill zero if field data is null
        cabin = []
        for s in dataset['Cabin']:
            cabin.append(ConvertCabinEncoder(subStr, convertTable))
        dataset['Cabin'] = cabin
        #dataset['Cabin'] = preprocessing.LabelEncoder().fit_transform(dataset['Cabin'])
        """
        return dataset

    def queryTrainDataInfo(self):
        traindatacount = len(self.TRAIN_DATA_SET)
        testdatacount = len(self.TEST_DATA_SET)
        positivedatacount = len(self.PositiveTrainDataSet)
        negativedatacount = len(self.NegativeTrainDataSet)
        print("Total Count :" + str(positivedatacount + negativedatacount))
        print("    Positive Data Count :" + str(positivedatacount))
        print("    Negative Data Count :" + str(negativedatacount))
        print("    Train Data Count (Total) : %4d" % (traindatacount ))
        print("    Test Data Count (Total): %4d" % (testdatacount))


    def setTrainData(self,traindata, traindataratio = 0.5, testfulldata = False):
        # split survived dataset or not survived dataset
        self.PositiveTrainDataSet = traindata[traindata['Survived'] == 1]
        self.NegativeTrainDataSet = traindata[traindata['Survived'] == 0]
        self.TrainDataRatio = traindataratio

        if traindataratio > 1.0 or traindataratio <= 0.0:
            self.TrainDataRatio = 1

        if len(self.PositiveTrainDataSet) < len(self.NegativeTrainDataSet):
            traindatacount = int(len(self.PositiveTrainDataSet) * self.TrainDataRatio)
        else:
            traindatacount = int(len(self.NegativeTrainDataSet) * self.TrainDataRatio)

        self.AGE_NORMALIZATION = traindata.Age.max() - traindata.Age.min()
        self.FARE_NORMALIZATION = traindata.Fare.max() - traindata.Fare.min()
        self.SIBSP_NORMALIZATION = traindata.SibSp.max() - traindata.SibSp.min()
        self.PARCH_NORMALIZATION = traindata.Parch.max() - traindata.Parch.min()

        if testfulldata:
            self.TRAIN_DATA_SET = pd.concat([self.PositiveTrainDataSet, self.NegativeTrainDataSet], ignore_index=True)
        else:
            self.TRAIN_DATA_SET = pd.concat([self.PositiveTrainDataSet.iloc[0:traindatacount,:], self.NegativeTrainDataSet.iloc[0:traindatacount,:]],ignore_index = True)

        if testfulldata or self.TrainDataRatio == 1.0:
            self.TEST_DATA_SET = pd.concat([self.PositiveTrainDataSet, self.NegativeTrainDataSet], ignore_index = True)
        else:
            self.TEST_DATA_SET = pd.concat([self.PositiveTrainDataSet.iloc[traindatacount:, :], self.NegativeTrainDataSet.iloc[traindatacount:, :]],ignore_index = True)

        self.TRAIN_LABEL_SET = pd.DataFrame(self.TRAIN_DATA_SET["Survived"]) #.values.ravel()
        self.TRAIN_DATA_SET = self.TRAIN_DATA_SET.drop(['Survived'], axis=1)  # drop axis = 1 (column)
        self.TRAIN_DATA_SET = self._NormalizationFeature(self.TRAIN_DATA_SET)

        self.TEST_LABEL_SET = pd.DataFrame(self.TEST_DATA_SET['Survived']) #conert serias to dataframe
        self.TEST_DATA_SET = self.TEST_DATA_SET.drop(['Survived'], axis=1)
        self.TEST_DATA_SET = self._NormalizationFeature(self.TEST_DATA_SET)

    """
    precision = TP / (TP + FP)  # 預測為正的樣本中實際正樣本的比例
    recall = TP / (TP + FN)  # 實際正樣本中預測為正的比例
    accuracy = (TP + TN) / (P + N)
    F1 - score = 2 / [(1 / precision) + (1 / recall)]
    """
    def traincrossSVMModel(self, tuned_parameters, score_parameters , cross_validation):
        for score in score_parameters:
            print ("Scoring Type : " + score)
            self.clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=cross_validation,
                        scoring="roc_auc")
                        #scoring='%s_macro' % score)

            self.clf.fit(self.TRAIN_DATA_SET, self.TRAIN_LABEL_SET.values.ravel())
            print("Best parameters set found on development set:")
            print(self.clf.best_estimator_)
            print(self.clf.best_params_)
            print("=== Train Data Set ===")
            precision_score = self.clf.score(self.TRAIN_DATA_SET, self.TRAIN_LABEL_SET.values.ravel())
            print(" Precision Score : ", precision_score)

            precision_cross_validation_score = cross_val_score(self.clf, self.TRAIN_DATA_SET.values,
                                                               self.TRAIN_LABEL_SET.values.ravel(), cv=4,
                                                               scoring='accuracy')
            print("=== Precision Cross Validation Score ===")
            print(" MEAN : %f (STD : %f) " % (
            precision_cross_validation_score.mean(), precision_cross_validation_score.std()))
            print(precision_cross_validation_score)

            receiver_operation_characteristic_curve = cross_val_score(self.clf, self.TRAIN_DATA_SET.values,
                                                                      self.TRAIN_LABEL_SET.values.ravel(), cv=4,
                                                                      scoring='roc_auc')
            print("=== Receiver Operation Characteristic Score ===")
            print(" MEAN : %f (STD : %f)" % (
            receiver_operation_characteristic_curve.mean(), receiver_operation_characteristic_curve.std()))
            print(receiver_operation_characteristic_curve)

        # C=20 best
    def trainSVMMode(self):
        self.clf = svm.SVC(kernel='rbf',C=2, gamma=0.002)
        self.clf.fit(self.TRAIN_DATA_SET, self.TRAIN_LABEL_SET.values.ravel())
        print("=== Train Data Set ===")
        precision_score = self.clf.score(self.TRAIN_DATA_SET, self.TRAIN_LABEL_SET.values.ravel())
        print(" Precision Score : ", precision_score)

        precision_cross_validation_score = cross_val_score(self.clf, self.TRAIN_DATA_SET.values,
                                                           self.TRAIN_LABEL_SET.values.ravel(), cv=4,
                                                           scoring='accuracy')
        print("=== Precision Cross Validation Score ===")
        print(" MEAN : %f (STD : %f) " % (
        precision_cross_validation_score.mean(), precision_cross_validation_score.std()))
        print(precision_cross_validation_score)

        receiver_operation_characteristic_curve = cross_val_score(self.clf, self.TRAIN_DATA_SET.values,
                                                                  self.TRAIN_LABEL_SET.values.ravel(), cv=4,
                                                                  scoring='roc_auc')
        print("=== Receiver Operation Characteristic Score ===")
        print(" MEAN : %f (STD : %f)" % (
            receiver_operation_characteristic_curve.mean(), receiver_operation_characteristic_curve.std()))
        print(receiver_operation_characteristic_curve)

    def testSVMModel(self):
        print ("=== Test Data Set ===")
        precision_score = self.clf.score(self.TEST_DATA_SET, self.TEST_LABEL_SET.values.ravel())
        print(" Precision Score : ", precision_score)

        precision_cross_validation_score = cross_val_score(self.clf,self.TEST_DATA_SET.values, self.TEST_LABEL_SET.values.ravel(), cv=4, scoring='accuracy')
        print("=== Precision Cross Validation Score ===")
        print(" MEAN : %f (STD : %f) " % (precision_cross_validation_score.mean(), precision_cross_validation_score.std()))
        print(precision_cross_validation_score)

        receiver_operation_characteristic_curve = cross_val_score(self.clf, self.TEST_DATA_SET.values, self.TEST_LABEL_SET.values.ravel(), cv=4, scoring='roc_auc')
        print("=== Receiver Operation Characteristic Score ===")
        print(" MEAN : %f (STD : %f)" % (receiver_operation_characteristic_curve.mean(), receiver_operation_characteristic_curve.std()))
        print(receiver_operation_characteristic_curve)

    def predictSVMMode(self, testPredict):
        predictDataSet = self._NormalizationFeature(testPredict)
        outpredict = self.clf.predict(predictDataSet.values)
        results = pd.DataFrame({
            'PassengerId': testDataSet['PassengerId'],
            'Survived': outpredict
        })
        results.to_csv("titanic_svm_rbf_scale_festure_7_C100_AvagTrain.csv", index=False)

#exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = SCRIPT_DIR +"/input/train.csv"
#print (TRAIN_FILE)
np.set_printoptions(linewidth=320)
pd.set_option('display.width',320)
pd.set_option('display.max_column',20)
pd.set_option('display.max_row',420)

# load all train
trainDataSet = pd.read_csv(TRAIN_FILE)
ppc = TitnaicPredictClass()
ppc.setTrainData(trainDataSet, 1, testfulldata=True)
ppc.queryTrainDataInfo()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2e-1, 2e-2, 2e-3, 2e-4],
                     'C': [0.2, 2, 20, 200, 2000]},
                    {'kernel': ['linear'], 'C': [0.2, 2, 20, 200, 2000]}]

#tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000] }]

scores = ['precision', 'recall']
#ppc.traincrossSVMModel(tuned_parameters, scores, 3)
ppc.trainSVMMode()


TEST_FILE = SCRIPT_DIR +"/input/test.csv"
testDataSet = pd.read_csv(TEST_FILE)
ppc.predictSVMMode(testDataSet)

print("=== Test Parameter ==")
ppc.trainSVMMode()
ppc.predictSVMMode(testDataSet)
exit(1)

# Test Function
group = trainDataSet[["Parch", "Survived"]].groupby(['Parch'])
print(group.describe())
group = trainDataSet.groupby(['Survived'])
print (group['Age'].describe())
print (group['Fare'].describe())

#print(pd.concat([ppc.TRAIN_DATA_SET, ppc.TRAIN_LABEL_SET],axis=1))
#print(pd.concat([ppc.TEST_DATA_SET, ppc.TEST_LABEL_SET],axis=1))

plt.figure()
selected = trainDataSet[trainDataSet['Survived'] == 1]
plt.hist(x=selected['Age'], bins=20, rwidth=0.5)
plt.show()
