import pandas as pd
import matplotlib.pylab as plt
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import itertools
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Permutation - Permutation entropy of time series: extract the sequence pattern of time series
def permutation_entropy(time_series, m, delay):
    """Calculate the Permutation Entropy"""
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    c = [0] * len(permutations)
    for i in range(n - delay * (m - 1)):
        # sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
        for j in range(len(permutations)):
            if abs(permutations[j] - sorted_index_array).any() == 0:
                c[j] += 1
    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe

# Extract data from Batch
def data_extract(i):
    x = []; y = [] ; permu = []
    path = ['/Users/mengkaixu/Desktop/ModellingQuestion/ModellingQuestion/ForInterview_Batch']
    path.append(str(i))
    form = '.xlsx'
    path.append(form)
    batch = pd.read_excel(''.join(path))
    batch = batch[1:]
    batch['LOD'].fillna(0, inplace=True)
    # nonzero index
    non_ind = list(np.nonzero(batch.values[:, 5])[0])
    if len(non_ind) > 1:
        non_ind.remove(0)
    for j in non_ind:
        sample = []; pervec = []
        diff = batch.values[j,0] - batch.values[0,0]
        days_to_hours = diff.days * 24
        diff_btw_two_times = (diff.seconds) / 3600
        diff_days = (days_to_hours + diff_btw_two_times)/24
        sample.append(diff_days)
        sub_batch = list(batch[:j+1].mean()[0:4].values)
        #x.append(sample + sub_batch)
        y.append(batch.values[j,5])
        ba = batch[:j + 1].values[:,1:5]
        for k in range(ba.shape[1]):
            pe = permutation_entropy(ba[:,k],3,1)
            pervec.append(pe)
        x.append(sample + sub_batch + pervec)
        permu.append(pervec)
    return x, y, permu

# Multi-layer Neural Network
def mlp(X,Y):
    model = Sequential()
    model.add(Dense(15, activation='relu',input_dim=XX.shape[1]))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
    model.fit(X, Y, epochs=2000, verbose=0)
    return model

def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

# Define curve (line) fitting functiona
def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

# Model performance measure - Mean Squared Error (MSE)
def err(yhat,ytest):
    a = 0
    for i in range(len(yhat)):
    a += (yhat[i] - ytest[i])**2
    MSE = a/len(yhat)
    print('Mean Squared Error %.3f' % MSE)
    return MSE

# Plot moisture data vs time step
color = ['ko','co','ro','bo','yo']
batchind = ['Batch1','Batch2','Batch3','Batch4','Batch5' ]
for i in range(5):
    b = i+1
    path = ['/Users/mengkaixu/Desktop/ModellingQuestion/ModellingQuestion/ForInterview_Batch']
    path.append(str(b))
    form = '.xlsx'
    path.append(form)
    batch = pd.read_excel(''.join(path))
    batch = batch[1:]
    plt.plot(batch['LOD'],color[i],label=batchind[i])
plt.xlabel('Time step',fontsize=18)
plt.ylabel('Moisture content',fontsize=18)
plt.legend(loc='upper right', borderaxespad=0.,fontsize=14)
plt.title('LOD Moisture content vs time',fontsize=22)

# Feature extraction & Training data preparation
train = []; Y = []; permut = []
for i in range(5):
    a = i+1
    x, y, permu = data_extract(a)
    train.append(x)
    Y.append(y)
    permut.append(permu)

# Reshape the extracted data
X=array(train)
X = X.reshape(X.shape[0]*X.shape[1],X.shape[2])
Y = array(Y)
Y = Y.reshape(Y.shape[0]*Y.shape[1])
permut = array(permut)
permut = permut.reshape(permut.shape[0]*permut.shape[1],permut.shape[2])

# Correlation Analyses between extracted features and  moisture response
cor = []
for i in range(9):
    corr, _ = pearsonr(X[:,i],Y)
    cor.append(round(corr,2))

# Scatter Plot between extracted features and  moisture response
fig, axs = plt.subplots(3, 3)
fig.suptitle('Scatter plot for each feature & moisture content', fontsize = 22)
axs[0,0].plot(X[:,0],Y,'o');axs[0,1].plot(X[:,1],Y,'o');axs[0,2].plot(X[:,2],Y,'o');
axs[1,0].plot(X[:,3],Y,'o');axs[1,1].plot(X[:,4],Y,'o');axs[1,2].plot(X[:,5],Y,'o');
axs[2,0].plot(X[:,6],Y,'o');axs[2,1].plot(X[:,7],Y,'o');axs[2,2].plot(X[:,8],Y,'o');
feature = ['Dring time','Pot Temperature','Jacket Temperature','Agitation Speed','Vessel Pressure',
           'PE Pot temperature', 'PE Jacket temperature','PE Agitation Speed','PE Vessel pressure']
for i in range(len(axs.flat)):
    axs.flat[i].set(xlabel=feature[i], ylabel='Moisture Content')


# Feature selection based on the ranking of correlation coefficient
XX = np.concatenate((X[:,0:2],X[:,6:7]), axis=1)

# Validation data preparation
test_batch_ind = [6, 7, 9, 11]
Test = []; y_test = []
for i in test_batch_ind:
    x, y, permu_test = data_extract(i)
    Test.append(x)
    y_test.append(y)
X_test=array(Test)
X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1],X_test.shape[2])
Y_test = array(y_test)
Y_test = Y_test.reshape(Y_test.shape[0]*Y_test.shape[1])
testX = np.concatenate((X_test[:,0:2],X_test[:,5:6]), axis=1)

"""Multi-layered Neural Network"""
# Multi-layer Neural Network (MLP): traning on batch 1-5, and predict the end moisture of Batch 6, 7, 9 & 11
Error = []
for i in range(20):
    model = mlp(XX,Y)
    yhat = model.predict(testX, verbose=0)
    error, _ = model.evaluate(testX, Y_test, verbose=0)
    Error.append(error)
    print(error)

"""Linear Regression"""
## Linear Regression (MLR): traning on batch 1-5, and predict the end moisture of Batch 6, 7, 9 & 11
regressor = LinearRegression()
regressor.fit(XX, Y)
yhat_linear = regressor.predict(testX)
error_linear = err(yhat_linear, Y_test)

"""Power-law fitting"""
## Power-law prediction (PL): traning on batch 1-5, and predict the end moisture of Batch 6, 7, 9 & 11
# Fit data using power-law
popt, pcov = curve_fit(func_powerlaw, X[:,0], Y, p0 = np.asarray([-1,10**5,0]))
yhat_pl = func_powerlaw(X_test[:,0], *popt)
error_pl = err(yhat_pl, Y_test)

# Plot fitted power-law curve, moisture content (training), moisture content (validation)
plt.figure()
plt.plot(np.arange(0.05, 1.1, 0.01), func_powerlaw(np.arange(0.05, 1.1, 0.01), *popt), 'k-', label="Fitted Function")
plt.plot(X[:,0], Y, 'bo', label="Training Batch")
plt.plot(X_test[:,0],Y_test,'ro', label="Validation Batch")
plt.xlabel('Drying time',fontsize=18)
plt.ylabel('Moisture Content',fontsize=18)
plt.legend(loc='upper right', borderaxespad=0.,fontsize=14)
plt.title('Power-law fitting',fontsize=22)

## Comparision between three methods
Over_MSE = {}
Over_MSE['MLP'] = Error
Over_MSE['MLR'] = error_linear
Over_MSE['PL'] = error_pl

color_dict = {'MLP':'orange', 'MLR':'blue', 'PL':'green'}
controls = ['MLP', 'MLR', 'PL']

fig, ax = plt.subplots()

boxplot_dict = ax.boxplot(
    [Over_MSE[x] for x in ['MLP', 'MLR', 'PL']],
    positions = [1, 1.5, 2],
    labels = controls,
    patch_artist = True,
    widths = 0.25)
plt.title('Model Comparision-MSE',fontsize=22)
plt.ylabel('MSE',fontsize=18)

ax.set_ylim([0.1,0.22])