import sys
import ast
import math
import random
import numpy as np
import pickle

def sigmoid(num): return 1/(1+math.exp(-num))

def sigmoidPrime(num): return sigmoid(num) * (1 - sigmoid(num))

def p_net(A, x, w_list, b_list):
    new_a = np.vectorize(A)
    a = list()
    a.append(np.array(list(x)))
    for l in range(1, len(w_list) + 1): a.append(new_a((a[l-1]@w_list[l-1]) + b_list[l-1]))
    return a[len(w_list)]

def backProp(e2, epochs, f, fP, trainingSet, w, b, lam):
    AFunct = np.vectorize(f)
    APrime = np.vectorize(fP)
    for e in range(e2, epochs):
        print(e)
        for z in trainingSet:
            #print(z)
            dot = [None]
            x, y  = z
            a = []
            a.append(x)
            for l in range(1, len(w)):
                dot.append((a[l-1]@w[l] + b[l]))
                a.append(AFunct(dot[l]))
            n = len(dot) - 1
            delta = [None for i in range(0, n)]
            delta.append((APrime(dot[n]))*(y-a[n]))
            for l in range(n - 1, 0, -1): delta[l] = (APrime(dot[l]))*(delta[l+1]@(w[l+1].transpose()))
            for l in range(1, len(w)):
                b[l] = b[l] + (lam*delta[l])
                w[l] = w[l] + (lam*((a[l-1].transpose())@delta[l]))
        f = open('mnistSave.pkl', 'wb')
        save = {'epoch': e + 1, 'trainSet': trainingSet, 'wList': w, 'bList': b}
        pickle.dump(save, f)
        f.close()
    return [w[1:], b[1:]]
print('(N)ew run or (C)ontinue old run?')
ans = 'c'
if str(ans) == 'N' or str(ans) == 'n':
    trainingSet = []
    wList1 = 2 * np.random.rand(784, 300) - 1
    wList2 = 2 * np.random.rand(300, 100) - 1
    wList3 = 2 * np.random.rand(100, 10) - 1
    bList1 = 2 * np.random.rand(1, 300) - 1
    bList2 = 2 * np.random.rand(1, 100) - 1
    bList3 = 2 * np.random.rand(1, 10) - 1
    mainWList = [None, wList1, wList2, wList3]
    mainBList = [None, bList1, bList2, bList3]
    epochNum = 0
    with open("mnist_train.csv") as f:
        line_list = [[int(x) for x in line.strip().split(',')] for line in f]
        for line in line_list:
            lList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            label = line[0]
            lList[label] = 1
            image = line[1:]
            trainingSet.append((np.array([image])/255, np.array([lList])))
else:
    f = open('mnistSave.pkl', 'rb')
    saved = pickle.load(f)
    f.close()
    epochNum = saved['epoch']
    trainingSet = saved['trainSet']
    mainWList = saved['wList']
    mainBList = saved['bList']
back = backProp(epochNum, 15, sigmoid, sigmoidPrime, trainingSet, mainWList, mainBList, 0.1)
w = back[0]
b = back[1]
print(w, b)
count2 = 0
for z in trainingSet:
    x, y = z
    p = p_net(sigmoid, x, w, b)
    count = 0
    i = 0
    #print(z)
    max2 = p[0][0]
    #print(max2)
    for x2 in p[0]:
        #print(x2)
        if max2 < x2: 
            max2 = x2
            i = count
        count += 1
    if y[0][i] != 1: count2 += 1
print("Percentage of misclassified items in the training set:", (count2 / len(trainingSet)))
count2 = 0
testSet = []
with open("mnist_test.csv") as f:
    line_list = [[int(x) for x in line.strip().split(',')] for line in f]
    for line in line_list:
        lList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label = line[0]
        lList[label] = 1
        image = line[1:]
        testSet.append((np.array([image])/255, np.array([lList])))
for z in testSet:
    x, y = z
    p = p_net(sigmoid, x, w, b)
    count = 0
    i = 0
    #print(z)
    max2 = p[0][0]
    #print(max2)
    for x2 in p[0]:
        #print(x2)
        if max2 < x2: 
            max2 = x2
            i = count
        count += 1
    if y[0][i] != 1: count2 += 1
print("Percentage of misclassified items in the test set:", (count2 / len(testSet)))

# [array([[-0.44787695, -0.24465156,  0.22492565, ..., -0.43298863,
#         -0.35366381,  0.07664401],
#        [ 0.27289657, -0.58086862, -0.53583691, ...,  0.4535884 ,
#         -0.86236851, -0.54541159],
#        [-0.64955014, -0.99966965, -0.05171447, ...,  0.72377789,
#         -0.58738142,  0.30070669],
#        ...,
#        [-0.80000798, -0.39641239,  0.73335333, ...,  0.494004  ,
#         -0.07181684,  0.92872019],
#        [ 0.06026934, -0.18583038, -0.43282494, ..., -0.98521573,
#         -0.24616848,  0.01748694],
#        [ 0.30745644,  0.73002347,  0.69900646, ..., -0.62223034,
#         -0.60976049,  0.78316828]]), array([[ 0.16680867, -0.28908588, -0.08658487, ..., -0.24541287,
#         -0.22437424, -0.35251301],
#        [-0.68436932,  0.3501346 , -1.19105633, ...,  0.92946987,
#          0.35629502, -0.84546662],
#        [-0.47173844, -0.40814247,  0.51108948, ...,  0.64826432,
#          0.7814898 , -0.21609228],
#        ...,
#        [ 0.77898497,  0.52817476, -0.86870727, ...,  0.71524669,
#         -0.7262859 , -0.24179998],
#        [ 0.58783638,  0.32347632, -0.28960721, ...,  0.37459233,
#         -0.52714575,  0.60510175],
#        [ 0.8599655 ,  0.33685998,  0.21545735, ...,  0.30353813,
#          0.52955687,  0.8763987 ]]), array([[ 1.17672228e-01, -6.99547738e-01,  1.64189879e+00,
#          1.72691770e+00,  5.81375778e-01, -2.30604586e+00,
#         -6.61859105e-01, -1.20000779e+00, -2.04830406e-01,
#          2.07356047e-01],
#        [ 9.87761550e-01,  1.49220176e+00, -2.44835187e+00,
#          4.01866557e-01, -1.06449418e+00, -3.24636978e-01,
#          6.44176886e-01, -2.19346058e-01, -1.89164499e-01,
#          1.53659148e-01],
#        [ 4.99962761e-01,  4.52005121e-01,  1.14334818e+00,
#          7.07964854e-01, -5.61271031e-01,  1.61546568e+00,
#         -7.34940560e-01, -1.26024201e+00, -1.15252188e+00,
#         -1.59446290e+00],
#        [ 5.78875840e-01, -1.10105769e+00, -6.25903305e-01,
#          1.77452484e+00,  2.07636540e-01, -2.48121146e+00,
#          1.08412949e+00, -4.55951103e-01,  1.24324642e+00,
#         -7.68733964e-01],
#        [-7.23952568e-01,  1.19572368e+00,  1.85547459e-02,
#          2.08261266e+00, -1.10625381e+00, -8.78932273e-02,
#          1.49759959e-01, -2.05749300e+00, -7.80854548e-01,
#         -1.08220677e+00],
#        [ 7.29411840e-01, -1.04960326e+00,  2.31858416e+00,
#         -6.24129176e-01, -2.63714447e-01, -3.30889108e-01,
#         -9.85465121e-01, -9.94257675e-01, -4.71836670e-03,
#          1.64622753e-01],
#        [ 1.41838812e+00, -1.06422153e+00,  8.90732779e-01,
#         -1.56300361e+00, -1.18329724e+00, -4.72781190e-01,
#          1.95620549e-01, -1.24193825e+00, -4.72317241e-01,
#          5.14224929e-01],
#        [ 3.22899806e-01,  4.16553979e-01, -2.77541319e+00,
#          1.92277106e+00,  1.20778044e+00, -1.25669533e+00,
#          1.07404480e+00,  3.20478070e-01,  2.14788215e+00,
#          3.45824067e-01],
#        [ 1.83435900e+00, -6.22180920e-01, -6.07781099e-01,
#         -4.09697541e-02,  9.74936373e-02, -3.11235955e+00,
#         -1.04324950e+00,  2.73949040e-01,  1.32727952e+00,
#         -1.32856778e+00],
#        [ 1.88884912e+00,  9.51112040e-01, -3.82042679e-01,
#         -1.20396531e+00,  9.28742325e-01,  1.94796631e+00,
#          2.03102611e-01,  5.61579527e-01, -3.99187562e+00,
#         -4.04552525e-02],
#        [-9.09218220e-01,  9.48622857e-01, -3.45106345e-01,
#         -8.35222226e-01,  2.31021875e+00,  5.81047777e-02,
#          5.88516390e-01,  3.12878524e+00, -2.79455671e+00,
#         -1.59587988e+00],
#        [ 1.80870166e-01, -7.11172155e-01,  8.16392941e-02,
#         -2.75551837e+00,  1.89974809e+00,  7.89438376e-01,
#         -1.79378923e+00,  1.93848663e+00, -2.82623644e+00,
#          9.38982624e-01],
#        [-2.33945949e-01,  6.25811168e-01, -9.56653027e-01,
#         -1.51996601e+00,  1.31539897e+00,  3.34984087e-01,
#          1.47030466e+00, -5.69824402e-02, -4.46599308e-01,
#         -7.75258986e-01],
#        [-1.90038755e-01, -1.15855040e+00, -1.37597756e+00,
#         -5.71519214e-01, -1.22416861e+00, -9.50397833e-01,
#          8.46280289e-02,  2.18529231e-01, -7.22377396e-01,
#         -4.73398578e-03],
#        [-7.84967589e-01,  1.23472429e+00,  9.49616932e-01,
#          4.06996298e-01, -3.02604812e+00, -5.75487724e-01,
#          1.37133024e+00, -1.09257785e+00, -1.38801586e+00,
#          1.54590113e+00],
#        [-2.05782422e-01,  5.55135279e-02,  3.39929200e-01,
#         -6.43737523e-01, -3.76454206e-01, -1.56140463e-01,
#          2.47129416e-01, -1.00352926e-01, -7.80497033e-01,
#         -1.41802263e-01],
#        [ 2.01540859e+00, -6.30042703e-01, -1.54328819e+00,
#          7.44428972e-01, -5.55906597e-02,  8.96225416e-01,
#         -4.28843642e-01, -3.40709963e-01, -1.12848103e+00,
#          2.12872232e-01],
#        [ 6.07987428e-02,  1.03946168e+00,  1.88675212e-01,
#         -9.93112990e-01,  8.77195024e-01, -7.62215005e-01,
#          7.82273571e-01,  6.01194513e-01, -1.42497273e+00,
#         -5.50820085e-01],
#        [ 9.96798475e-01,  6.49688435e-01, -5.07036798e-01,
#         -7.76109238e-01,  5.84520126e-01,  3.28408148e-01,
#         -8.60405224e-02, -1.75735037e-01, -1.13956114e+00,
#          4.56943395e-01],
#        [-8.66622229e-01, -8.02204622e-01, -5.37590248e-02,
#         -1.31600815e+00, -9.72447232e-01, -1.18155604e+00,
#          3.25230706e-01, -9.10117749e-01, -6.42032744e-01,
#         -1.91583859e-01],
#        [-8.39519444e-01, -7.99035693e-01,  1.86466788e+00,
#          5.63394918e-01, -2.54463814e+00,  1.01831349e+00,
#         -1.45002733e+00,  1.71641057e+00, -8.79711405e-01,
#         -1.27949343e+00],
#        [-7.82804106e-01,  4.58249830e-01,  3.54202789e-01,
#         -5.25544079e-01,  2.81216011e-01, -5.20286849e-01,
#         -2.13347897e-01,  1.28041378e-01, -5.82215455e-01,
#         -1.83384647e-01],
#        [ 9.53103275e-01, -1.26136489e-01, -7.59238670e-02,
#          1.75278516e-01, -3.96717046e-01, -4.41715793e-01,
#         -8.18428729e-02,  7.82710151e-01, -6.13292386e-01,
#          2.38427439e-01],
#        [-7.07360147e-01, -6.67268915e-01, -6.83609667e-01,
#         -6.95023244e-01, -1.31202132e+00, -5.94154578e-01,
#         -2.52250857e-01,  6.08390023e-02, -8.10683365e-01,
#         -6.84427488e-01],
#        [-1.13262260e+00, -1.37940369e+00,  2.52986413e+00,
#          9.14827407e-01, -4.13109381e-01, -3.32629480e-01,
#         -2.42717690e+00,  1.09838154e+00,  8.59518609e-01,
#          5.59675496e-01],
#        [-1.90109844e-01, -1.44572637e+00, -9.65702477e-01,
#         -2.34844488e+00,  5.22583893e-01,  2.14712448e+00,
#         -2.34196179e+00, -1.20440248e+00,  6.21941061e-01,
#          1.49515461e-01],
#        [ 1.22698477e+00, -1.02417526e+00, -1.71157535e+00,
#          1.09584561e+00,  6.32159770e-01, -1.24604175e-01,
#          5.07646398e-01, -1.56214268e+00,  1.20866611e+00,
#          5.53270570e-01],
#        [-2.77234246e+00,  1.24338564e-01, -2.88628118e-01,
#          9.99104313e-01, -1.02742246e+00,  2.05729656e+00,
#          2.42176908e+00, -1.29300846e+00,  7.12952712e-01,
#          1.12987845e+00],
#        [ 8.43128621e-02, -1.74309103e+00,  3.79672687e-01,
#         -2.60987197e-01, -2.98063202e+00, -6.08685964e-02,
#         -1.43636690e+00, -1.94452936e-01, -1.14561703e-01,
#          2.25476827e+00],
#        [-1.73104764e+00, -3.97929071e-01, -6.67938532e-01,
#         -1.51962754e-01,  2.48303867e+00, -6.09332191e-01,
#         -1.12845243e-01,  1.82356461e+00, -1.36586745e+00,
#         -4.09506057e+00],
#        [-5.74810435e-01, -3.90158126e-01, -1.73276855e+00,
#          2.35575367e+00, -4.51141412e-01,  1.23596114e+00,
#         -8.33670773e-01,  3.91813405e-01, -2.11563073e-01,
#          1.51005900e+00],
#        [-3.85547189e-01, -1.15091102e+00,  9.00077272e-02,
#          1.50110849e+00, -6.96852560e-01,  1.25664744e+00,
#         -1.00463633e+00,  1.12899245e+00, -1.13255652e-01,
#         -1.31603652e+00],
#        [ 4.55515732e-01, -5.09589041e-01, -1.51822813e-01,
#         -6.32912630e-02,  4.65021492e-01,  7.14793831e-01,
#          1.09303939e-02,  6.95641900e-01, -5.97740225e-01,
#          1.92842100e-01],
#        [-5.61070365e-01, -7.38887504e-01,  6.43978649e-02,
#          9.11243561e-01,  4.78300221e-01, -2.87714236e-01,
#          1.17589213e-01,  7.44158709e-01,  3.13712168e-01,
#          3.56564136e-01],
#        [ 7.89971755e-01, -7.35645308e-01,  5.69981595e-01,
#          4.41197597e-01,  1.39925620e+00,  2.05030874e+00,
#          6.00047854e-01, -3.75800015e-01, -2.15797586e+00,
#         -2.39921540e+00],
#        [-1.17918832e+00, -1.51057567e+00,  8.84752648e-02,
#         -6.57207072e-01,  2.34186898e+00,  3.68341491e-01,
#          1.21939297e+00, -2.46480554e+00, -2.89970425e+00,
#          3.50259704e+00],
#        [ 1.57798368e+00, -6.40647987e-01, -1.08448881e+00,
#         -9.57064948e-01,  2.08822645e-02,  1.16142788e+00,
#          3.37929345e-01,  9.27041266e-01, -7.83696045e-01,
#         -2.27779196e-01],
#        [ 5.06401057e-01, -8.28738596e-01,  2.05203277e-01,
#         -1.48480652e+00,  8.90910875e-01, -6.66218950e-01,
#         -6.12693073e-02, -1.35823924e+00,  7.71100215e-01,
#          3.39931489e-01],
#        [-1.86358120e+00,  8.51576543e-01,  1.58691211e+00,
#          6.53832784e-01,  1.79676547e+00, -7.12113196e-01,
#         -1.62645818e+00, -4.11791359e-02, -1.86644952e+00,
#         -1.50651676e+00],
#        [-6.80048398e-01,  9.22089107e-01, -2.39878375e+00,
#         -1.68096667e+00, -2.15188282e-01, -1.94644364e-01,
#         -7.14595475e-01,  1.85673878e+00,  1.93807998e-01,
#          3.50459523e-01],
#        [-1.45736858e+00, -5.96450445e-01, -9.61780500e-02,
#         -5.83820127e-01, -1.52575667e+00, -5.16303820e-02,
#         -1.38768822e+00,  1.96243463e+00,  7.62540343e-01,
#         -8.30902758e-01],
#        [-9.01800966e-01, -1.88015953e+00, -6.92711609e-01,
#         -3.02694203e+00, -5.76429246e-01,  1.52777690e+00,
#          8.98791553e-01, -1.30987651e+00,  1.48234872e+00,
#         -1.39491467e+00],
#        [ 1.04707719e+00,  1.34723321e+00, -1.72895242e+00,
#         -5.75841186e-01,  3.64123032e-01, -3.85061056e-01,
#         -8.31893750e-02,  7.27237966e-01,  4.38802873e-01,
#         -1.98910424e+00],
#        [-1.33145616e+00,  7.80059766e-01,  5.85696049e-01,
#          7.02901344e-01, -2.86466112e+00,  7.99713917e-01,
#          1.96579342e+00, -1.23935056e+00, -1.59366178e+00,
#          1.17303000e+00],
#        [-1.37007487e+00,  1.66091856e+00, -1.16880260e+00,
#          1.00672451e+00, -2.56162721e-01,  1.36677192e-01,
#         -1.94426503e+00,  1.54424029e+00,  1.41004782e+00,
#          8.40946209e-01],
#        [ 1.11517007e+00, -1.46003001e-02,  1.44468317e+00,
#          7.39319373e-01, -8.82863562e-01, -8.84293403e-01,
#          6.43217085e-01, -4.42370628e-01, -2.30556968e+00,
#         -9.95253115e-01],
#        [ 4.80246957e-01,  3.84398042e-01, -2.41754635e-01,
#         -7.30577554e-03,  8.81970376e-02, -1.40992437e+00,
#         -1.32906225e+00, -4.54517736e-01, -8.56248274e-01,
#         -1.28466338e+00],
#        [ 1.55550377e+00, -1.66169360e+00,  2.06413926e+00,
#         -1.66766775e+00, -1.89566961e+00, -1.81956233e+00,
#         -1.62529645e+00, -1.96540582e+00, -5.80845951e-02,
#          1.46497139e+00],
#        [-5.54535263e-01, -9.71770642e-01, -1.25892409e+00,
#         -1.74149765e-01, -1.97109859e+00, -1.29546935e+00,
#          1.38147005e+00, -5.64997725e-02,  1.05073900e+00,
#          1.43201200e+00],
#        [ 6.47300350e-01,  8.98806327e-01, -2.78984764e-01,
#         -4.17145185e-01, -8.37056796e-01, -5.46982353e-01,
#         -1.24886891e+00, -2.37806837e+00, -3.05250154e-01,
#          3.03165021e+00],
#        [-1.09268414e+00,  1.26781554e+00,  6.93743261e-01,
#         -1.96345571e+00, -6.63990420e-01, -8.74623666e-02,
#          1.20477148e+00,  1.09883023e+00,  1.64178253e+00,
#         -1.90759211e+00],
#        [-1.28424736e-01, -9.06186279e-01, -1.96327436e+00,
#          1.00551125e+00,  7.14011391e-01, -1.37791395e+00,
#          1.23051931e+00,  8.29303164e-01,  1.80835306e+00,
#         -2.51811029e+00],
#        [ 1.30959356e-01, -1.59097369e-01, -6.74354140e-02,
#         -3.33823858e-01, -8.34881757e-01,  2.53347808e-01,
#         -1.56450315e+00, -5.28285660e-01, -9.45522574e-01,
#         -6.81003415e-01],
#        [-7.00700835e-01, -1.63441299e+00, -1.01008498e+00,
#         -2.30529532e+00, -9.99276296e-01,  3.39101132e+00,
#          1.04969279e+00, -6.05202105e-01,  1.32181995e+00,
#          1.95863469e-01],
#        [-1.36688543e+00, -1.27844510e+00, -2.99249741e+00,
#          2.47208505e+00, -6.46190100e-01,  1.74825442e+00,
#         -1.56299326e+00,  1.77099112e+00, -3.99550381e+00,
#          3.25272135e-01],
#        [ 5.44330174e-01,  6.52204304e-01, -2.83366884e-01,
#          1.75840044e+00,  1.95409673e+00, -1.79769058e+00,
#         -1.92334581e+00,  2.05054175e-01,  1.00738074e+00,
#         -9.30090095e-01],
#        [-6.41325529e-01, -8.39797207e-01,  1.12477210e+00,
#         -1.48317049e+00,  5.70751541e-01,  1.22657011e+00,
#         -4.89167633e-01, -4.92648757e-01,  1.64859052e-01,
#         -1.54483209e+00],
#        [-2.36234824e-01, -2.13928107e-01, -9.50619640e-01,
#          1.54509856e+00,  2.77099573e-01, -1.63572532e+00,
#          1.06206595e+00, -4.21678226e-01,  5.59788360e-01,
#          1.16119778e-01],
#        [ 5.89608412e-01,  5.82682976e-01, -7.99461980e-01,
#          1.10250448e+00, -1.17342880e+00,  2.51512435e+00,
#          1.15985022e+00, -9.44174208e-01,  2.39747639e+00,
#         -1.81292050e+00],
#        [-1.24227360e+00, -8.39918268e-01, -9.98226944e-01,
#         -8.70852558e-01,  6.78313407e-02, -3.30916487e-01,
#          6.29323834e-02, -7.79405748e-01,  2.34851011e-01,
#         -8.48409857e-01],
#        [-1.22341794e+00,  5.87870638e-01,  2.18801423e-01,
#         -6.87055550e-01, -1.16423173e+00, -1.43395226e+00,
#          2.91880760e-01,  4.34748066e-01, -1.15358974e+00,
#         -1.14849399e+00],
#        [ 2.01981443e+00, -5.88635343e-01,  1.04561551e+00,
#          1.95653093e+00, -2.73193577e+00, -2.11811644e+00,
#         -4.81757558e-01,  1.28070827e+00, -1.95000013e+00,
#          2.32511868e+00],
#        [-6.32142406e-01,  4.13824353e-01, -1.90960727e-01,
#         -6.74743686e-03,  2.82320560e-01,  2.93360624e-01,
#          2.01377448e-01, -6.59076725e-01,  2.73601853e-01,
#          5.35801357e-01],
#        [-8.87002466e-01,  1.58084351e+00, -5.56669049e-01,
#         -1.88304850e+00, -8.23406586e-01,  1.48943107e+00,
#         -1.07673768e+00, -1.37970340e+00,  1.50828614e-01,
#          1.36615902e+00],
#        [-1.54939538e+00, -1.36472282e+00, -1.42488685e+00,
#          3.09401821e-01,  5.28167235e-01,  1.16725622e+00,
#         -9.81401003e-01,  1.17703933e+00,  1.67886627e+00,
#         -8.04956234e-01],
#        [ 2.77442565e-01, -9.71419643e-03, -2.36014425e-01,
#         -7.07041853e-01,  5.71211530e-02,  4.09097240e-01,
#         -3.80670605e-01,  1.72431618e+00,  5.99304610e-01,
#         -1.99194863e+00],
#        [-1.23801806e+00,  1.80760541e+00, -1.05676247e+00,
#         -9.02522380e-01, -2.54578605e-02,  1.36292751e-01,
#         -3.91477587e-01,  7.04809127e-01,  4.03052150e-01,
#          6.93381855e-01],
#        [-8.45557957e-01, -1.79469785e-01, -7.85058638e-01,
#         -8.57128143e-01,  2.40629648e-01, -1.62741706e+00,
#         -1.01770091e+00, -4.74389583e-02, -1.49796041e+00,
#         -6.22785565e-01],
#        [-4.61082687e-01,  3.73996596e-02,  2.50171870e+00,
#         -1.29960706e+00, -6.60245875e-01,  1.71324505e+00,
#          3.74400068e-01, -6.90544617e-01, -7.25522418e-01,
#         -1.21496572e+00],
#        [ 5.77927476e-01,  6.11119747e-02,  3.33213710e-01,
#          1.03232930e+00, -6.72004834e-01, -2.03983308e+00,
#          1.36719024e+00, -1.95452378e-01,  5.21768354e-01,
#         -2.12746232e+00],
#        [ 9.19135630e-01,  1.34471476e+00, -3.10500446e-01,
#         -2.11101470e+00, -2.88437590e-01, -3.08004145e-01,
#         -9.80828139e-01,  1.07393385e+00, -4.15399771e-01,
#          1.42584023e+00],
#        [-1.50330069e+00, -2.14466992e+00,  9.16881406e-01,
#         -2.32772689e-01,  1.66549527e+00, -2.24140396e+00,
#          2.16894745e-01,  1.14142552e+00,  1.64952324e+00,
#         -7.19450792e-01],
#        [ 9.28281414e-01,  1.57019140e+00,  9.09082338e-02,
#         -2.07050622e+00,  7.73477178e-01, -8.55868209e-01,
#          6.07899139e-01,  1.88770370e+00, -1.92194002e+00,
#         -1.61400371e+00],
#        [ 3.94983153e-01,  5.19008999e-01,  8.04030436e-01,
#          1.30038117e-01, -7.84918333e-01,  7.80547805e-01,
#         -3.02838341e-01, -2.91684462e-01,  6.41785037e-01,
#          4.56279343e-02],
#        [ 3.71740079e-01, -1.42372945e-01, -2.00742369e+00,
#          4.08480436e-01,  1.42041481e+00, -7.37120280e-01,
#         -5.38627420e-01,  1.39209934e+00,  1.57090747e+00,
#          1.00481369e+00],
#        [-3.01714787e-01, -5.70636762e-01,  5.56831851e-01,
#          7.46596850e-01,  1.25029239e+00, -9.96137164e-01,
#         -6.50569276e-01, -2.12199816e+00, -3.21537725e-01,
#          1.87174734e+00],
#        [ 2.27072888e+00,  5.43733933e-01, -3.99280349e-01,
#         -1.53733546e-02, -1.70684665e+00, -1.18740110e+00,
#         -6.30852301e-01, -2.33319059e-01,  5.17788399e-01,
#         -4.57472992e-01],
#        [-5.08733854e-02, -1.03052155e+00, -8.76760334e-01,
#          9.40481130e-01,  5.43321345e-01,  8.16354518e-01,
#          4.59777185e-01,  5.27793133e-02, -8.38241748e-01,
#          5.68133234e-01],
#        [-2.87297978e-01, -3.78938279e-01, -1.25757025e+00,
#         -1.21240120e+00, -1.30415744e+00, -1.17101954e-01,
#         -1.19765652e+00, -9.18336672e-02,  1.73662822e-01,
#         -7.60093834e-01],
#        [-1.01040894e+00, -7.34717322e-01,  3.04785961e-01,
#         -9.58974542e-03, -1.13541565e+00, -1.36885124e+00,
#         -2.77347789e-02,  1.52152650e+00,  6.59480027e-01,
#         -6.57107749e-01],
#        [-1.75901170e+00, -7.74369455e-01, -1.53813299e+00,
#          3.69819341e-01, -1.47850012e+00,  9.90796590e-01,
#         -4.90798529e-01,  8.38155274e-01, -7.10978091e-01,
#         -4.93640792e-02],
#        [-6.74417071e-01, -1.71704518e-01, -5.10983531e-01,
#          2.06109988e+00, -9.87456176e-01, -9.61476175e-01,
#         -1.05700167e+00, -5.12920663e-01, -1.90472025e+00,
#          7.20876837e-01],
#        [ 6.25366610e-01, -2.06991671e-01,  2.21911979e-01,
#         -2.58264321e-01, -2.91990172e-01,  3.99700490e-01,
#         -1.05957302e+00, -2.53682015e+00, -7.22885073e-01,
#          3.65288342e-01],
#        [ 3.07415475e-01, -2.91131970e-01,  6.97753957e-01,
#          4.48214175e-01,  1.88313833e-01,  9.84197803e-01,
#          3.90475020e-01, -4.32349679e-02,  5.20028140e-01,
#          1.60211137e-01],
#        [ 3.63163793e-01, -1.74276285e+00,  6.62029054e-02,
#         -4.83585226e-01,  1.28717218e+00, -1.60371033e-01,
#          7.68923714e-01, -8.73721424e-01,  1.55334023e+00,
#         -1.11944462e+00],
#        [ 4.04815657e-01,  1.35409269e+00, -1.16437303e+00,
#         -1.21877023e+00,  6.39743354e-01,  1.15250645e+00,
#         -3.46489196e-01, -5.17839343e-01, -1.50138295e+00,
#         -2.19817152e-02],
#        [-1.00739834e+00,  8.11536110e-01, -6.46468855e-01,
#         -2.12169492e+00,  3.60756990e-01, -6.89121441e-03,
#         -2.20579840e-01, -8.99174193e-01,  8.16581061e-01,
#          6.04093817e-01],
#        [-2.53041914e+00,  1.07879986e+00,  4.75622179e-01,
#         -5.54074485e-01, -1.19053330e+00,  7.55187692e-01,
#          1.51943275e+00,  1.36501768e-01,  1.35815856e+00,
#         -1.23543647e+00],
#        [-2.04666290e-03, -1.02544219e+00, -2.51812331e+00,
#         -1.17554868e+00, -2.55503067e-01,  5.75164481e-01,
#          1.69495226e+00, -8.53777108e-01, -1.14715992e+00,
#         -5.53949984e-01],
#        [ 1.11971418e-01, -8.64091232e-01,  1.54289892e+00,
#         -1.43924643e+00, -6.08314606e-01,  2.03948273e+00,
#          1.52242264e-01, -7.74788584e-01,  1.13200554e+00,
#          2.10520402e-01],
#        [ 3.80790922e-01, -1.02046906e+00,  1.16563154e+00,
#          1.89672670e+00,  1.14872788e-01,  2.75482582e+00,
#         -2.74089033e+00, -9.93536948e-02, -3.18494163e+00,
#         -1.56657656e+00],
#        [ 5.85769592e-01,  1.96293435e-02,  9.51622360e-01,
#         -5.43334854e-01,  1.14083816e+00,  2.79920743e+00,
#         -2.91201768e+00, -1.17587341e+00,  1.33140198e+00,
#         -5.27909761e-01],
#        [-1.03723636e+00, -1.62803571e+00,  1.60233317e+00,
#         -1.62986011e+00, -1.44483019e+00,  1.99686545e+00,
#         -1.06219324e+00, -1.72171441e-01, -1.57124169e-01,
#         -1.66808735e+00],
#        [-9.09748568e-01,  1.70904204e+00,  2.96925707e-01,
#         -3.37918902e-01,  1.09080635e+00, -1.61439432e+00,
#          2.70780281e-01, -1.40061391e+00, -1.60304344e-01,
#         -8.52743232e-01],
#        [-8.01006158e-01,  9.33837425e-01,  1.82349424e+00,
#         -1.35444472e+00, -5.69298069e-01,  1.70032515e-01,
#         -1.12150408e+00,  1.25931529e+00, -7.05710324e-01,
#         -2.41789759e+00],
#        [-4.83827958e-01, -4.72360394e-01, -1.77124722e-01,
#         -3.74451750e-01, -1.13284065e+00,  2.94994940e-02,
#         -5.85517968e-01,  2.73252127e-01, -1.56398840e-01,
#          1.60538292e-01],
#        [-9.85967470e-01, -4.02305299e-01, -4.41901376e-01,
#         -1.03559159e+00, -1.47971066e+00, -1.63890462e+00,
#         -9.19017002e-01,  3.65785449e-01, -1.03955413e+00,
#          6.72155086e-03],
#        [ 3.32662279e-01, -5.53814833e-01, -1.34345490e+00,
#          6.04260616e-01, -2.33879817e-02, -8.53384945e-01,
#          6.13820194e-01,  5.36147109e-01, -3.23665728e-01,
#          5.64141968e-01],
#        [-1.08528973e+00, -4.72060357e-01,  1.40042244e+00,
#          1.61098701e-01, -2.36057196e-01, -8.04902304e-02,
#          2.86733869e-02,  4.96105442e-01, -5.77628684e-01,
#          2.02534762e-01],
#        [-1.32929461e-01,  6.26921491e-01, -2.33132300e-01,
#         -5.36100570e-01, -8.34541069e-01,  1.79777258e-01,
#          5.54524787e-02,  5.88059878e-02,  6.51111341e-01,
#          5.80787335e-01]])] [array([[-1.12553598e+00, -1.59332077e-01,  1.30583784e-01,
#          1.06597590e+00, -5.28740600e-01,  7.13564388e-02,
#         -1.15819156e+00,  1.48368572e-01, -1.03247729e+00,
#         -1.12448595e+00,  9.49306449e-01,  4.41829129e-01,
#         -7.42602048e-01, -7.27847645e-01, -6.80105431e-01,
#          4.95771269e-02, -4.26825692e-01, -9.78591462e-01,
#          3.56513264e-01,  6.62093368e-01, -8.99128940e-01,
#         -7.54577836e-02,  6.44909642e-01, -2.11271006e-01,
#         -7.39603617e-01,  4.03770557e-01, -1.26976635e+00,
#         -2.11695725e-01, -6.82318280e-01,  5.08152981e-01,
#          6.19026567e-01, -4.01333261e-01,  8.87529766e-02,
#          7.84118500e-01, -4.71855258e-02,  7.30954903e-02,
#         -8.29666280e-02,  3.26580181e-01, -1.38013038e-01,
#         -7.99695138e-01, -8.86020087e-01, -9.25751372e-01,
#         -6.07542557e-01, -6.65581305e-01, -1.54381361e-01,
#         -4.50311512e-01,  5.78282477e-01, -1.08732551e-02,
#         -1.00895272e+00,  1.00332427e+00, -2.65440459e-01,
#         -4.58279111e-01,  4.62784250e-01,  3.60012170e-01,
#          2.80199520e-01,  6.21481730e-01,  6.35307709e-01,
#         -9.03397627e-01,  8.01395721e-01,  5.24862204e-02,
#         -4.81878561e-01, -5.97420121e-01, -1.48605839e-01,
#          6.66516762e-01, -9.44249230e-01,  5.74499079e-01,
#         -1.10573163e+00, -7.17009331e-01,  7.23767071e-01,
#          7.77615034e-01, -4.89623682e-01,  1.09307306e+00,
#          7.59429620e-02, -3.27392007e-01,  1.04940106e+00,
#         -4.86437460e-01,  1.30212428e+00,  7.19468095e-01,
#          6.63586723e-01,  7.09521563e-01, -6.21531834e-01,
#         -2.41613991e-01, -2.90724692e-01, -5.93124423e-01,
#          3.14474271e-01, -4.54297872e-01, -1.06719536e+00,
#         -6.98921386e-01, -5.50913383e-01,  4.08122143e-01,
#         -1.34964405e+00, -5.56692278e-01, -3.82641054e-01,
#         -3.63571556e-01,  7.66351251e-01, -6.91980764e-01,
#         -2.56026028e-01, -6.25240636e-01,  6.60491092e-01,
#          4.54958634e-01,  8.28320654e-01,  6.64516942e-01,
#          9.08314063e-01, -6.66596453e-01,  3.00323903e-02,
#          3.45158578e-01, -6.90812682e-01,  6.95160520e-01,
#         -2.67607397e-01, -1.02093660e-01, -5.97035598e-01,
#         -4.09596815e-01, -4.12773757e-01, -8.78255832e-01,
#         -4.63771959e-01,  5.19071468e-01, -3.67660319e-01,
#         -6.97725424e-01,  6.37151623e-01,  6.65122627e-01,
#          7.73541746e-01, -5.68116993e-01,  5.51925551e-02,
#          2.35389977e-01, -8.03326238e-01,  1.56243220e-01,
#          7.74021614e-01, -8.82950454e-01, -4.57124801e-01,
#          8.46618840e-01, -1.61232474e-02,  7.44403098e-01,
#         -4.95099707e-01, -4.18770071e-01, -7.50089395e-01,
#          4.57061256e-01, -6.09895843e-01,  5.99342511e-01,
#          1.50661314e-01, -1.84909067e-01,  8.23668483e-01,
#         -2.54060810e-01, -5.50892383e-01, -3.49282006e-01,
#         -1.02772802e-01,  7.17876956e-01, -9.39566745e-01,
#         -8.12120157e-01,  5.86389816e-01,  3.38399067e-02,
#         -1.03978835e+00,  7.01776112e-01,  1.20736699e+00,
#         -4.19362015e-01, -4.30424125e-01,  1.16849925e+00,
#          4.25244369e-01,  8.83837953e-01,  3.03500429e-01,
#         -3.92431066e-01,  8.02167881e-01,  3.89404602e-01,
#         -1.02202885e-01, -1.12667907e-01,  6.49332927e-01,
#         -2.84549540e-02, -1.19190040e+00,  2.44446871e-01,
#          7.11500922e-01,  6.26768229e-02, -3.20591881e-01,
#         -2.73261401e-01, -5.48036029e-01, -6.20713504e-02,
#          4.10374647e-01,  7.68300433e-01, -5.46620149e-01,
#         -8.05453662e-01,  3.69543265e-01, -4.60684292e-01,
#          2.35913896e-01, -6.71282409e-01,  3.03594353e-01,
#          4.52620744e-01, -8.35931576e-01, -1.01403973e+00,
#          4.91972350e-01, -6.80130540e-01,  2.08311480e-01,
#          1.16510682e-01, -3.01665832e-01, -9.25687056e-01,
#         -9.89420678e-01, -6.79372922e-01,  1.07820881e+00,
#         -5.74752257e-01, -5.14761437e-01, -5.16197574e-01,
#          7.14334851e-02,  1.03504560e+00,  8.51951107e-01,
#         -5.71791628e-01, -5.28228862e-01,  8.95169832e-01,
#          7.19676516e-02, -3.78773544e-02, -4.35866228e-01,
#         -2.51935594e-01, -2.05056968e-02, -8.70072706e-01,
#         -6.57592273e-01,  5.79228257e-02,  9.55386075e-02,
#          3.70754685e-01, -4.00945791e-01,  7.96278212e-01,
#         -1.93601325e-01, -9.09957528e-01, -2.40182760e-01,
#          2.18966026e-01,  2.70056354e-01,  6.62147376e-01,
#         -6.47356044e-01,  2.39883167e-01, -5.57450404e-01,
#          5.73546022e-01,  8.50045073e-01, -8.69401189e-01,
#         -1.66796932e-01,  6.44641187e-01,  8.11035137e-01,
#          5.57513001e-01, -4.38832762e-01, -1.45752990e-01,
#          3.81954145e-01, -3.79573707e-01,  5.73684461e-01,
#         -7.26242949e-01,  1.60114919e-01,  2.35325294e-01,
#         -7.83074164e-02, -1.06778221e-04,  1.85378791e-01,
#          3.93723067e-01,  9.80010849e-02,  5.97497880e-01,
#         -3.45332840e-01,  2.43515310e-01,  9.66786284e-01,
#          3.51276043e-01, -9.76198869e-01, -6.68529481e-01,
#          6.58103735e-02, -9.96023796e-01,  3.14160110e-01,
#         -7.81659671e-01,  5.17108130e-01,  7.05042880e-01,
#         -8.07553102e-02,  5.49690745e-01, -4.03090703e-01,
#          2.00043509e-01, -8.80326169e-01,  7.16453185e-01,
#         -3.13937776e-01,  2.14126020e-01,  4.26911619e-01,
#          1.92139774e-02, -2.57333247e-01,  1.28485714e-01,
#          9.08717620e-01, -1.50282291e+00, -6.05594546e-01,
#          1.36147309e-01,  9.07912736e-01,  4.52154092e-01,
#         -2.96525902e-01, -1.14561060e+00,  4.69030101e-01,
#          6.96506572e-01, -7.66349281e-01, -8.12255711e-02,
#          5.76938101e-01, -5.46680111e-01, -3.12639895e-02,
#          7.65146568e-01, -5.33824754e-01,  1.22904205e+00,
#          3.51007036e-01, -2.56733830e-01, -8.39971359e-01,
#          4.25215390e-01,  9.11418436e-01,  4.33013820e-01,
#         -2.74392555e-01, -6.93091807e-02,  8.45669852e-01,
#         -2.73675831e-01, -8.41578707e-01,  4.29515822e-01]]), array([[-0.85682418, -0.06452652, -0.69057196, -0.0493315 , -0.87984974,
#          0.41565096,  0.69121631,  0.80053283,  0.65695008,  0.39523702,
#          0.0910165 , -0.32770791,  0.77818968, -0.33169293,  0.57305634,
#         -0.43406361, -0.85947574,  0.34073918, -0.86072981,  0.81711609,
#          0.09003365,  0.11398602,  0.18981833,  0.4651346 ,  0.22518005,
#         -0.2759745 , -0.76590117,  0.73402726, -0.76349184, -0.13303386,
#          0.76713789,  0.97238325, -0.63431739,  0.15546354, -0.99183929,
#         -0.2072433 , -0.88128857,  0.89662698, -0.06810311, -0.39719083,
#          0.75673191,  0.05649377, -1.02820724,  0.8774175 ,  0.9474402 ,
#         -0.1567807 , -0.08006587,  0.33991429, -0.36094738,  0.85919472,
#         -0.07402733,  0.83906446,  0.97569694,  0.68673587,  0.53451657,
#         -0.77131763, -1.00200763, -0.30679489,  0.06813549,  0.7366619 ,
#          0.15059629, -0.61222085, -0.34551713,  0.29616433,  0.24019564,
#          0.91799412, -0.64328884, -0.35391394,  0.33971722, -0.5083221 ,
#          0.32816337, -0.93188678, -0.21156948, -0.91436572, -0.60301269,
#         -0.59784439, -0.9539808 ,  0.41686763,  0.1323012 ,  0.16285071,
#          0.7076099 , -0.41043263,  0.13062006, -0.47339067,  0.42643541,
#          0.72660549,  0.05034902,  0.58186412,  0.76214282,  0.31315756,
#          0.33134972,  0.78010024, -0.40139655, -0.61067026, -0.88415243,
#         -0.56742978,  0.04846896,  0.51259153, -0.83619008, -0.87670568]]), array([[ 0.1746583 , -0.07341547, -0.07929275, -1.11137127,  0.26567643,
#         -0.00582613, -0.81909355,  0.10201831, -0.8586798 ,  0.21504431]])]