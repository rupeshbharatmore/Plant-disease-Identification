





def function(inp):

    import numpy as np
    import pandas as pd
    from skimage import feature, io
    from sklearn import preprocessing
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    import cv2

    b = []
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True



    img = io.imread(inp, as_gray=True)

    S = preprocessing.MinMaxScaler((0,11)).fit_transform(img).astype(int)
    Grauwertmatrix = feature.greycomatrix(S, [1,2,3], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=12, symmetric=False, normed=True)




    ContrastStats = feature.greycoprops(Grauwertmatrix, 'contrast')
    CorrelationtStats = feature.greycoprops(Grauwertmatrix, 'correlation')
    HomogeneityStats = feature.greycoprops(Grauwertmatrix, 'homogeneity')
    ASMStats = feature.greycoprops(Grauwertmatrix, 'ASM')


    a = [np.mean(ContrastStats),np.mean(CorrelationtStats),np.mean(ASMStats),np.mean(HomogeneityStats)]
    b.append(a)



    dict1 = {1:"bacterialspot",2:"Healthy",3:"Lateblight",4:"tomato_mosaic",5:"yellowcurved",6:"anyther"}

    dataset = pd.read_csv('fun.csv', encoding='utf-8')

    X = dataset.iloc[:,[0,1,2,3]].values

    y = dataset.iloc[:,4].values





    from sklearn.model_selection import train_test_split

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)




    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    x_train = sc.fit_transform(x_train)

    x_test  = sc.transform(x_test)

    #print(x_test)







    '''from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors = 5,p = 2)

    classifier.fit(x_train,y_train)'''



    #hear enter healthy.jpeg
    from sklearn.svm import SVC # "Support Vector Classifier" 
    clf = SVC(kernel='sigmoid') 
    
    # fitting x samples and y classes 
    clf.fit(x_train, y_train) 

    y_pred = clf.predict(b)

    #hear enter download.jpeg
    '''from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5)

    clf.fit(x_train,y_train)

    y_pred = clf.predict(b)

    y_pred1 = clf.predict(x_test)

    print("accuracy:"+str(accuracy_score(y_test,y_pred1)*100))'''




    '''
    clf = LogisticRegression(solver = 'lbfgs')

    clf.fit(x_train,y_train)

    y_pred = clf.predict(b)'''

    








    

    return (dict1[y_pred[0]])

b = function(input("enter you image"))
print(print('given plant is of type having disease' +'  ' +b))
