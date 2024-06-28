import numpy as np

from sklearn.model_selection import train_test_split
def create_demo():
    np.random.seed(34)

    Xs1 = np.random.normal(loc=1,scale=0.5,size=(300,5))
    Ys1 = -2*Xs1[:,0]+1*Xs1[:,1]-0.5*Xs1[:,2]

    Xs2 = np.random.normal(loc=-1,scale=0.5,size=(300,5))
    Ys2 = -0.5*Xs2[:,2]+1*Xs2[:,3]-2*Xs2[:,4]
    X_data = np.concatenate((Xs1, Xs2), axis=0)
    Y_data = np.concatenate((Ys1.reshape(-1, 1), Ys2.reshape(-1, 1)), axis=0)
    Y_data = Y_data-Y_data.min()
    Y_data=Y_data/Y_data.max()
    case_labels = np.concatenate((np.array([1] * 300), np.array([2] * 300)))
    Y_data = np.concatenate((Y_data, case_labels.reshape(-1, 1)), axis=1)
    return X_data,Y_data

    X_train, X_remain, yc_train, yc_remain = train_test_split(X_data, Y_data, train_size=0.8, shuffle=True,
                                                              random_state=34)
    X_valid, X_test, yc_valid, yc_test = train_test_split(X_remain, yc_remain, train_size=0.5, shuffle=True,
                                                          random_state=34)


