def neuronTest(id, clas):
    place = np.dot(trainVectors[id], hyperPlanes[id]) - shifts[id]
    for i in range(60000):
        if i == id:
            continue
        if ((clas == trainLabels[i] and place * (np.dot(trainVectors[i], hyperPlanes[id])-shifts[id]) < 0)
        or (clas != trainLabels[i] and place * (np.dot(trainVectors[i], hyperPlanes[id])-shifts[id]) > 0)):
            return False
        
    return True

def calculateHyper(points): #calculates hyperplane coefficients
    if np.linalg.det(points) == 0:
        u, s, vh = np.linalg.svd(points)
        return vh[-1]
    else:
        one = np.ones((784,1))
        return np.matrix.dot(np.linalg.inv(points), one)

def patternTest(pattern): # test each neurons for each class
    for i in range(len(neuronByClass)):
        flag = 0
        for neurons in neuronByClass[i]:
            place = np.dot(trainVectors[neurons], hyperPlanes[neurons]) - offsets[neurons]
            if (np.dot(pattern, hyperPlanes[neurons]) - shifts[neurons]) * place < 0:
                flag = 1
                break
        if flag == 0:
            return i
    return -1