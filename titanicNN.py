import pandas as pd
import numpy as np
import tensorflow as tf

train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

#print(train_data.describe(include="all"))

#print(train_data.columns)

#for item in train_data.columns:
#    print(item, train_data[item].isnull().sum())   

titles = []
testTitles = []

# Extract the title from the name
for item in train_data['Name']:
    titles.append(item[(int)(item.find(',')+2):(int)(item.find('.'))])

for item in test_data['Name']:
    testTitles.append(item[(int)(item.find(',')+2):(int)(item.find('.'))])

# get the counts of each title

titlesUnique = set(titles)
testTitlesUnique = set(testTitles)

titleCounts = {}

# find counts of each title
for item in titlesUnique:
    titleCounts[item] = (titles.count(item))

# print the counts of each title
for item in titleCounts:
    print(item, titleCounts[item])
# Mr has 517, Miss has 182, Mrs has 125, Master has 40, and the rest have less than 10

# If the title isn't one of the main four we can just call it other
for item in titles:
    if(item != 'Mr' and item != 'Miss' and item != 'Mrs' and item != 'Master'):
        titles[titles.index(item)] = 'Other'

for item in testTitles:
    if(item != 'Mr' and item != 'Miss' and item != 'Mrs' and item != 'Master'):
        testTitles[testTitles.index(item)] = 'Other'

# Age guesses based on title, the "Other"s are probably mostly older. Could probably make better guesses but this is a start
AgeGuesses = []

titleTotals = {"Mr": 0, "Miss": 0, "Mrs": 0, "Master": 0, "Other": 0}

for item in titles:
    if(pd.isnull(train_data['Age'][titles.index(item)])):
        if(item == 'Mr'):
            AgeGuesses.append(22)
        elif(item == 'Miss'):
            AgeGuesses.append(26)
        elif(item == 'Mrs'):
            AgeGuesses.append(38)
        elif(item == 'Master'):
            AgeGuesses.append(2)
        else:
            AgeGuesses.append(40)
    else:
        AgeGuesses.append(train_data['Age'][titles.index(item)])
        titleTotals[item] += train_data['Age'][titles.index(item)]

titlesUnique = set(titles)

for item in titlesUnique:
    titleCounts[item] = (titles.count(item))

# Find the average age for each title
for item in titleTotals:
    print(item, titleTotals[item]/titleCounts[item])
        

# Do the same for the test data
testAgeGuesses = []
for item in testTitles:
    if(pd.isnull(test_data['Age'][testTitles.index(item)])):
        if(item == 'Mr'):
            testAgeGuesses.append(22)
        elif(item == 'Miss'):
            testAgeGuesses.append(26)
        elif(item == 'Mrs'):
            testAgeGuesses.append(38)
        elif(item == 'Master'):
            testAgeGuesses.append(2)
        else:
            testAgeGuesses.append(40)
    else:
        testAgeGuesses.append(test_data['Age'][testTitles.index(item)])

# Replace the ages with the guesses
train_data['Age'] = AgeGuesses

test_data['Age'] = testAgeGuesses

# If the cabin value is recorded they are more likely to survive
train_data["HasCabin"] = (train_data["Cabin"].notnull().astype('int'))
test_data["HasCabin"] = (test_data["Cabin"].notnull().astype('int'))

# Smaller families are more likely to survive
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1

# If they are alone they are more likely to survive
train_data["IsAlone"] = (train_data["FamilySize"] == 1).astype('int')
test_data["IsAlone"] = (test_data["FamilySize"] == 1).astype('int')

# There are only two values missing so we can just fill them with the most common value
train_data["Embarked"] = train_data["Embarked"].fillna('S')
test_data["Embarked"] = test_data["Embarked"].fillna('S')

# Convert the categorical values to numerical values
train_data["Embarked"] = train_data["Embarked"].map({'S': 0, 'C': 1, 'Q': 2}).astype('int')
train_data["Sex"] = train_data["Sex"].map({'female' : 0, 'male' : 1})

test_data["Embarked"] = test_data["Embarked"].map({'S': 0, 'C': 1, 'Q': 2}).astype('int')
test_data["Sex"] = test_data["Sex"].map({'female' : 0, 'male' : 1})

test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())

train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)
test_data = test_data.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)

print(train_data.describe(include="all"))
print(test_data.describe(include="all"))

modelMake = True
if(modelMake):

    y_train = train_data[0:891]['Survived'].values
    x_train = train_data[0:891].drop(['Survived'], axis=1).values

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(196, activation='relu', input_dim=8)) 
    model.add(tf.keras.layers.Dropout(0.5) )
    model.add(tf.keras.layers.Dense(196, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5) )
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5) )
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    # Compiling the NN
    model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Train the NN
    model.fit(x_train, y_train, batch_size = 64, epochs = 1000)

    model.save('titanicNN.h5')
else:    
    model = tf.keras.models.load_model('titanicNN.h5')

# Predict the test data
predictions = model.predict(test_data.drop(['PassengerId'], axis=1).values)

# Round the predictions to 0 or 1
rounded = []
for item in predictions:
    if(item >= 0.5):
        rounded.append(1)
    else:
        rounded.append(0)

# Create the submission file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': rounded})
submission.to_csv('submission.csv', index=False)
    
