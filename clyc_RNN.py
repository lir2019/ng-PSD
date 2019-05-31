import numpy as np
import tensorflow as tf
import scipy.io as scio
import matplotlib.pyplot as plt

INPUT_NODES=10240#总的输入节点数
INPUT_NODE=1024#每个时刻的输入节点数
NUM_STEPS=10#总时刻数，即RNN循环体单元数
STATE_SIZE = 50#每个RNN循环体单元的输出节点数

BATCH_SIZE=100#每轮训练的样本数

RATE = 0.01#学习率

CYCLE_TIMES=30001#循环次数

datafile1='train_in_set2.mat'
datafile2='train_out_set2.mat'
datafile3='test_in_set2.mat'
datafile4='test_out_set2.mat'
data1=scio.loadmat(datafile1)
data2=scio.loadmat(datafile2)
data3=scio.loadmat(datafile3)
data4=scio.loadmat(datafile4)
X_tr=data1['train_in']
Y_tr=data2['train_out']
X_te=data3['test_in']
Y_te=data4['test_out']


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

train_examples=np.shape(X_tr)[0]
test_examples=np.shape(X_te)[0]

X_tr=np.reshape(np.array(X_tr,dtype=np.float32),[train_examples,INPUT_NODES])
Y_tr=np.array(Y_tr,dtype=np.float32)
X_te=np.reshape(np.array(X_te,dtype=np.float32),[test_examples,INPUT_NODES])
Y_te=np.array(Y_te,dtype=np.float32)

def generate_batch(raw_data):
    raw_x, raw_y = raw_data
    dataset_size=np.shape(raw_x)[0]
    
    batchs=dataset_size//BATCH_SIZE
    for i in range(batchs):
        start=i*BATCH_SIZE
        end=min(start+BATCH_SIZE,dataset_size)
        x=raw_x[start:end,:]
        y=raw_y[start:end,:]
        yield (x,y)

#这里的n就是训练的循环次数
def generate_epochs(X_tr,Y_tr,n):
    for i in range(n):
        yield generate_batch((X_tr,Y_tr))

x0 = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_NODES])
y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 3])
x=tf.reshape(x0,[BATCH_SIZE, NUM_STEPS,INPUT_NODE])
#RNN的初始化状态，全设为零
init_state = tf.zeros([BATCH_SIZE, STATE_SIZE])
 
#将输入unstack，即在num_steps上解绑，方便给每个循环单元输入。
rnn_inputs = tf.unstack(x, axis=1)
 
#定义rnn_cell的权重参数，
with tf.variable_scope('rnn_cell'):
    #由于tf.Variable() 每次都在创建新对象，所有reuse=True 和它并没有什么关系。
    #对于get_variable()来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。
    W = tf.get_variable('W', [STATE_SIZE+INPUT_NODE, STATE_SIZE])
    b = tf.get_variable('b', [STATE_SIZE], initializer=tf.constant_initializer(0.0))
#使之定义为reuse模式，循环使用，保持参数相同
def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [STATE_SIZE+INPUT_NODE, STATE_SIZE])
        b = tf.get_variable('b', [STATE_SIZE], initializer=tf.constant_initializer(0.0))
    #定义rnn_cell具体的操作，这里使用的是最简单的rnn，激活函数为f(x)=tanh(x)
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
 
state = init_state
rnn_outputs = []

#循环num_steps次，即将一个序列输入RNN模型
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]
 
#定义softmax层
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [STATE_SIZE, 3])
    b = tf.get_variable('b', [3], initializer=tf.constant_initializer(0.0))
#注意，这里要将num_steps个输出全部分别进行计算其输出，然后使用softmax预测
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]
y=logits[-1]

loss=tf.reduce_mean((logits[-1]-y_)*(logits[-1]-y_))/2
correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(logits[-1],1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_step = tf.train.GradientDescentOptimizer(RATE).minimize(loss)
#train_step = tf.train.AdamOptimizer(RATE).minimize(loss)


count_times=0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for idx, epoch in enumerate(generate_epochs(X_tr,Y_tr,CYCLE_TIMES)):
        
        training_state = np.zeros((BATCH_SIZE, STATE_SIZE))

        for step, (X, Y) in enumerate(epoch):
            count_times+=1
            if count_times>CYCLE_TIMES:
                break
            sess.run(train_step,feed_dict={x0:X, y_:Y, init_state:training_state})
            if count_times%30000==1:
                    
                counts1=0
                counts2=0
                t_l1=0
                t_l2=0
                t_a1=0
                t_a2=0
                    
                for idx1, epoch1 in enumerate(generate_epochs(X_tr,Y_tr,1)):
                    for step1, (X1, Y1) in enumerate(epoch1):
                        loss1, a1= sess.run(
                            [loss,accuracy],
                            feed_dict={x0:X1, y_:Y1, init_state:training_state})
                        t_l1+=loss1
                        t_a1+=a1
                        counts1+=1

                for idx2, epoch2 in enumerate(generate_epochs(X_te,Y_te,1)):
                    for step1, (X2, Y2) in enumerate(epoch2):
                        loss2, a2= sess.run(
                            [loss,accuracy],
                            feed_dict={x0:X2, y_:Y2, init_state:training_state})
                        t_l2+=loss2
                        t_a2+=a2
                        counts2+=1
                print("%6d  %.6f    %.6f    %.6f    %.6f"%(count_times,
                    t_l1/counts1,t_l2/counts2,t_a1/counts1,t_a2/counts2))
 

    for i in range(50):
        data=scio.loadmat('clyc'+str(i+1)+'_2.mat')
        X=data['y2']
        m=(np.shape(X)[0]//BATCH_SIZE+1)*BATCH_SIZE
        X1=np.zeros((m,INPUT_NODES))
        Y=np.zeros((m,3))
        RNN_results=np.zeros((m,3))
        
        X1=np.reshape(np.array(X1,dtype=np.float32),[m,INPUT_NODES])
        Y=np.array(Y,dtype=np.float32)
        
        '''
        for j in range(np.shape(X)[0]):
            X1[j]=X[j]
    
        for idx, epoch in enumerate(generate_epochs(X1,Y,1)):
            training_state = np.zeros((BATCH_SIZE, STATE_SIZE))
            for step, (X, Y) in enumerate(epoch):
                y1= sess.run(y,
                    feed_dict={x0:X, y_:Y, init_state:training_state})
            
                RNN_results[step*BATCH_SIZE:(step+1)*BATCH_SIZE]=np.array(y1)

        scio.savemat('RNN_results'+str(i+1)+'.mat', {'RNN_results': RNN_results}) 
        '''


