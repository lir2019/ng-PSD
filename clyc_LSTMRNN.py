import numpy as np
import tensorflow as tf
import scipy.io as scio

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

INPUT_NODES=256#输入的总节点数
INPUT_NODE=32#每一个时刻输入的节点数
NUM_STEPS=8#总时刻数，即LSTM单元数
OUTPUT_NODES=3#输出节点数

HIDDEN_SIZE=50#LSTM结构的隐含层节点数
NUM_LAYERS=1#LSTM结构的层数

TRAINING_STEPS=30000#训练循环次数
BATCH_SIZE=100#每轮训练的样本数
RATE=0.01#学习率

#载入样本数据
def generate_data(is_training):
    
    datafile1='train_in_set2.mat'
    datafile2='train_out_set2.mat'
    datafile3='test_in_set2.mat'
    datafile4='test_out_set2.mat'
    data1=scio.loadmat(datafile1)
    data2=scio.loadmat(datafile2)
    data3=scio.loadmat(datafile3)
    data4=scio.loadmat(datafile4)
    X_tr1=data1['train_in']
    Y_tr1=data2['train_out']
    X_te1=data3['test_in']
    Y_te1=data4['test_out']
    
    X_tr1=np.array(X_tr1)[0:15000,:]
    Y_tr1=np.array(Y_tr1)[0:15000,:]
    X_te1=np.array(X_te1)[0:15000,:]
    Y_te1=np.array(Y_te1)[0:15000,:]
    
    X_tr=np.zeros((np.shape(X_tr1)[0],256));

    X_te=np.zeros((np.shape(X_tr1)[0],256));
    

    
    for i in range(256):
        for j in range(np.shape(X_tr1)[0]):
            X_tr[j,i]=sum(X_tr1[j,i*40:i*40+40])
            X_te[j,i]=sum(X_te1[j,i*40:i*40+40])
            
    Y_tr=Y_tr1
    Y_te=Y_te1

    '''
    datafile1='tr_i.mat'
    datafile2='tr_o.mat'
    datafile3='te_i.mat'
    datafile4='te_o.mat'
    data1=scio.loadmat(datafile1)
    data2=scio.loadmat(datafile2)
    data3=scio.loadmat(datafile3)
    data4=scio.loadmat(datafile4)
    X_tr=data1['tr_i']
    Y_tr=data2['tr_o']
    X_te=data3['te_i']
    Y_te=data4['te_o']
    '''
    
    training_examples=np.shape(X_tr)[0]
    testing_examples=np.shape(X_te)[0]
    

    if is_training:
        return np.reshape(np.array(X_tr,dtype=np.float32),[training_examples,INPUT_NODE,NUM_STEPS]), np.array(Y_tr,dtype=np.float32)
    else:
        return np.reshape(np.array(X_te,dtype=np.float32),[testing_examples,INPUT_NODE,NUM_STEPS]), np.array(Y_te,dtype=np.float32)

#定义LSTM单元
def lstm_model(X,y):
    #设置NUM_LAYERS可以改变LSTM结构的层数
    cell=tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])
    
    #将LSTM结构连接成RNN并计算其前向传播结果
    outputs, _ =tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)#outputs是顶层LSTM在每一步的输出结果，维度是[BATCH_SIZE,time,HIDDEN_SIZE]
    
    output=outputs[:,-1,:]#output为最后一个时刻LSTM的输出
    
    predictions=tf.contrib.layers.fully_connected(#对最后一个时刻LSTM的输出再加一层全连接层
        output,OUTPUT_NODES,activation_fn=None)
    
    loss=tf.losses.mean_squared_error(labels=y,predictions=predictions)#定义损失函数
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(predictions,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#定义分类正确率
    
    train_op=tf.contrib.layers.optimize_loss(#创建模型优化器并得到优化步骤
        loss,tf.train.get_global_step(),
        optimizer="Adagrad",learning_rate=RATE)
    return predictions,loss,accuracy,train_op
    
def train(sess,train_X,train_y,test_X, test_y):
    #将训练集数据以数据集的方式提供给计算图
    ds=tf.data.Dataset.from_tensor_slices((train_X,train_y))
    ds=ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X,y_=ds.make_one_shot_iterator().get_next()
    
    #调用模型，得到预测结果，损失函数，正确率和训练操作
    with tf.variable_scope("model"):
        predictions, _, _,train_op=lstm_model(X,y_)
    
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS+1):
        if i%2000==0:
            
            ds=tf.data.Dataset.from_tensor_slices((train_X, train_y))
            ds=ds.batch(1)
            X1, y1_=ds.make_one_shot_iterator().get_next()
    
            with tf.variable_scope("model", reuse=True):
                prediction, loss1, _, _=lstm_model(X1, y1_)

            acc1=0        
            total_l1=0;    
            for j in range(np.shape(train_X)[0]):
                p,l1,y1=sess.run([prediction,loss1,y1_])
                if np.where(p==np.max(p))==np.where(y1==np.max(y1)):
                    acc1+=1
                total_l1+=l1;
            
            ds=tf.data.Dataset.from_tensor_slices((test_X, test_y))
            ds=ds.batch(1)
            X2, y2_=ds.make_one_shot_iterator().get_next()
    
            with tf.variable_scope("model", reuse=True):
                prediction, loss2, _, _=lstm_model(X2, y2_)
                
            acc2=0
            total_l2=0;
            for j in range(np.shape(test_X)[0]):
                p,l2, y2=sess.run([prediction,loss2, y2_])
                if np.where(p==np.max(p))==np.where(y2==np.max(y2)):
                    acc2+=1
                total_l2+=l2

            
            #每训练500次打印一次训练集上的损失函数，训练集上的分类正确率，测试集上的损失函数，测试集上的分类正确率
            print("%6d  %.6f    %.6f    %.6f    %.6f"%(i+1,
                total_l1/np.shape(train_X)[0],total_l2/np.shape(test_X)[0],
                acc1/np.shape(train_X)[0],acc2/np.shape(test_X)[0]))
        
        if i==TRAINING_STEPS:
            break
        
        #执行优化步骤
        sess.run(train_op)
    
    for i in range(50):
        data=scio.loadmat('clyc'+str(i+1)+'_2.mat')
        X3=data['y2']
        
        X3=np.array(X3)[:,:]
        X4=np.zeros((np.shape(X3)[0],256));

        for k in range(256):
            for j in range(np.shape(X3)[0]):
                X4[j,k]=sum(X3[j,k*40:k*40+40])

        
        X3=X4
        
        
        number=np.shape(X3)[0]
        X3=np.reshape(np.array(X3,dtype=np.float32),[number,INPUT_NODE,NUM_STEPS])
        Y3=np.zeros((number,3))
        Y3=np.array(Y3,dtype=np.float32)
        
        ds=tf.data.Dataset.from_tensor_slices((X3,Y3))
        ds=ds.batch(1)
        X4,Y4=ds.make_one_shot_iterator().get_next()
        with tf.variable_scope("model", reuse=True):
            prediction, _, _, _=lstm_model(X4, Y4)
        LSTM_results=np.zeros((number,3))
        for j in range(number):
            LSTM_result=sess.run(prediction)
            LSTM_results[j]=LSTM_result

        scio.savemat('LSTM_results'+str(i+1)+'.mat', {'LSTM_results': LSTM_results}) 
    

train_X, train_y=generate_data(True)

test_X, test_y=generate_data(False)


with tf.Session() as sess:
    train(sess, train_X, train_y,test_X, test_y)
