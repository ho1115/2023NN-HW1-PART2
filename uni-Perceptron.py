import math
import numpy as np
import json
import time

def typeHandler(it):
    if isinstance(it, np.integer):
        return int(it)
    elif isinstance(it, np.floating):
        return float(it)
    elif isinstance(it, np.ndarray):
        return it.tolist()
    else:
        return it

def shiftPlane(points, plane, index): # step3, shifts hyperplane
    if len(points) == 0:
        shifts.append(offsets[index])
        return
    shortest = abs(np.dot(trainVectors[points[0]], plane) - offsets[index])
    id = points[0]
    for i in range(1, len(points)): # determines closest pattern to hyperplane in Pn2
        newDis = abs(np.dot(trainVectors[points[0]], plane) - offsets[index])
        if newDis < shortest:
            shortest = newDis
            id = points[i]
    shifts.append((offsets[index] + np.dot(trainVectors[id], plane))/2) # doing the shift

def vectorDis(vec1, vec2): # calculates vector distance
    sum = 0
    for i in range(len(vec1)):
        sum += (vec1[i] - vec2[i]) ** 2
    return math.sqrt(sum)

def calculateHyper(points): # calculates hyperplane coefficients
    cent = np.mean(points, axis=0)
    u, s, vh = np.linalg.svd(points - cent)
    return (vh[-1], np.dot(vh[-1], cent)) # returns hyperplane weights and centroid

def CalculatePn1Pn2(index):
    tmpDict = {}
    vector = trainVectors[index]
    for i in range(60000):
        if trainLabels[i] == trainLabels[index]: # 同一類不用算distance
            continue
        tmpDis = vectorDis(vector, trainVectors[i])
        tmpDict[i] = tmpDis
    tmpDict = dict(sorted(tmpDict.items(), key = lambda item: item[1])[0 : 784]) # 前784近的不同類pattern
    with open('allPn1\\' + str(index) + '.json', 'w') as fp:
        json.dump(tmpDict, fp, default=typeHandler)
    passPoints = []
    tmpPn2 = []
    for k, v in tmpDict.items(): # group points to a matrix
        passPoints.append(trainVectors[k])
    res = calculateHyper(passPoints)
    coe = res[0]
    off = np.dot(res[1], coe) # calculate hyperplane offset by centroid
    offsets.append(off)
    flag = 0 # log which side is positive
    if np.dot(trainVectors[index], coe) >= off:
        flag = 1
    else:
        flag = -1

    sameLabel = sortByLabel[trainLabels[index]] # retrieve all patterns in the same class

    for j in range(len(sameLabel)): # making Pn2
        place = np.dot(trainVectors[sameLabel[j]], coe) - off
        if (flag == 1 and place > 0) or (flag == -1 and place < 0):
            tmpPn2.append(sameLabel[j])

    with open('Pn2' + str(index) + '.json', 'w') as fp:
        json.dump(tmpPn2, fp, default=typeHandler)
    
    with open('hyperPlane' + str(index) + '.json', 'w') as fp:
        json.dump(coe, fp, default=typeHandler)

def removePattern(ban, id): # removes overlayed patterns 
    for item in ban:
        if item in allPn2[id]:
            allPn2[id].remove(item)

# main 
trainVectors = []
trainLabels = []
sortByLabel = []

with open('BaseJson.json') as jsfile:
    js = json.load(jsfile)
    trainVectors = js['trainVectors']
    trainLabels = js['trainLabels']
    sortByLabel = js['sortByLabel']

offsets = [] # store original hyperplane offsets
for i in range(60000): # executes step1 and step2
    CalculatePn1Pn2(i)

allPn2 = []
hyperPlanes = []

for i in range(60000): # merge 60000 jsons to one json
    with open('Pn2' + str(i) + '.json', 'r') as fp:
        allPn2.append(fp)    
    with open('hyperPlane' + str(i) + '.json', 'r') as fp:
        hyperPlanes.append(fp)
    
with open('Pn2.json', 'w') as fp:
    json.dump(allPn2, fp, default=typeHandler)
with open('Hyper.json', 'w') as fp:
    json.dump(hyperPlanes, fp, default=typeHandler)



# with open('Hyper.json') as jsfile:
#     hyperPlanes = json.load(jsfile)

# with open('Pn2.json') as jsfile:
#     allPn2 = json.load(jsfile)

# with open('Offs.json') as jsfile:
#     offsets = json.load(jsfile)



# step 3 

shifts = []
for i in range(60000): # step 3
    shiftPlane(allPn2[i], hyperPlanes[i], i)

with open('Shifts.json', 'w') as fp:
    json.dump(shifts, fp, default=typeHandler)

# with open('Shifts.json') as jsfile:
#     shifts = json.load(jsfile)

neuronByClass = []
for i in range(10): #step 4
    group = sortByLabel[i]
    foundNeuron = []
    while (len(group) > 0) :
        group.sort(key = lambda index : len(allPn2[index]), reverse=True) # 照Pn2大小sort
        if len(allPn2[group[0]]) == 0:
            break
        maxPn2 = allPn2[group[0]]
        foundNeuron.append(group[0])     
        for j in range(1, len(group)): # 刪去其他Pn2中已出現的patterns
            removePattern(maxPn2, group[j])
        group.pop(0)
    neuronByClass.append(foundNeuron)

with open('result.json', 'w') as fp:
    json.dump(neuronByClass, fp, default=typeHandler)

# with open('result.json', 'r') as fp:
#     neuronByClass = json.load(fp)
    
# train result test
error = 0

right = [0,0,0,0,0,0,0,0,0,0] #紀錄每個class的正確次數
type1 = [0,0,0,0,0,0,0,0,0,0] #紀錄每個class的type 1 error次數
type2 = [0,0,0,0,0,0,0,0,0,0] #紀錄每個class的type 2 error次數
for clas in range(10): 
    for i in range(60000):
        neurons = neuronByClass[clas]
        flag = 1
        for item in neurons:
            place = np.dot(trainVectors[item], hyperPlanes[item]) - offsets[item] # 檢測test dataset accuracy時，此行依然需使用train dataset json
            if (np.dot(trainVectors[i], hyperPlanes[item]) - shifts[item]) * place < 0:
                flag = 0
                break
        if flag == 1:
            if trainLabels[i] == clas:
                right[clas] += 1
            else:
                type1[clas] +=1
        elif flag == 0:
            if trainLabels[i] == clas:
                type2[clas] += 1
            else:
                right[clas] += 1

for i in range(10):
    print('class' + str(i) + 'accuracy =', (right[i]/60000), 'right = ', right[i] ,'/', 60000, 'type1 error = ', type1[i] ,'/', 60000, 'type2 error = ', type2[i] ,'/', 60000)