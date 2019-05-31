import tensorflow as tf
import scipy.io as scio
import numpy as np

INPUT_NODES=10240#总的输入节点数
OUTPUT_NODES=3#输出节点数

CONV_SIZE=512#卷积核的长度
STRIDE=256#卷积层的步长
CONV_DEEP=20#卷积层的深度
FC_SIZE=50#第一个全连接层的节点数
BATCH_SIZE=100#每轮训练的样本数

CYCLE_TIMES=6000#训练的循环次数
RATE=0.001#学习率

#申请空间
x0=tf.placeholder(tf.float32,[None,INPUT_NODES])#placeholder定义存放样本数据的地方；None表示个数不确定
x=tf.reshape(x0,[-1,INPUT_NODES,1,1])#第一维表示样本数据个数，-1表示不确定；第二维第三维表示输入矩阵的尺寸；第四维表示矩阵的深度,也称通道数
y_=tf.placeholder(tf.float32,[None,OUTPUT_NODES])#第一维表示样本数据个数，None表示不确定；第二维输出节点数

#定义和初始化卷积核权重参数conv_w和偏置项参数conv_b
conv_w=tf.Variable(tf.random_normal([CONV_SIZE,1,1,CONV_DEEP],stddev=0.1,seed=1))#第一第二维表示卷积核的尺寸，第三维表示通道数，第四维表示卷积核的个数也称卷积层的深度
conv_b=tf.Variable(tf.random_normal([CONV_DEEP],stddev=0.1,seed=1))


#定义卷积层
conv=tf.nn.conv2d(x,conv_w,strides=[1,STRIDE,1,1],padding='SAME')#trides的第一第四维要求一定为1，第二第三维分别表示在矩阵长和宽方向上的步长，padding='SAME'表示全0填充
relu=tf.nn.relu(tf.nn.bias_add(conv,conv_b))#卷积层的激活函数选择为ReLU函数，即f(x)=max(x,0)

#将卷积层的各个数据拉成一个向量，作为全链接层的输入
relu_shape=relu.get_shape().as_list()
nodes=relu_shape[1]*relu_shape[2]*relu_shape[3]
relu_flat=tf.reshape(relu,[-1,nodes])

#定义和初始化第一个全连接层的权重参数fc1_w和偏置项参数fc1_b
fc1_w=tf.Variable(tf.random_normal([nodes,FC_SIZE],stddev=0.1,seed=1))
fc1_b=tf.Variable(tf.random_normal([FC_SIZE],stddev=0.1,seed=1))

#定义第一个全连接层
fc1=tf.nn.relu(tf.matmul(relu_flat,fc1_w)+fc1_b)#第一个全连接层的激活函数选择为ReLU函数，即f(x)=max(x,0)

#定义和初始化第二个全连接层（即输出层）的权重参数fc2_w和偏置项参数fc2_b
fc2_w=tf.Variable(tf.random_normal([FC_SIZE,OUTPUT_NODES],stddev=0.1,seed=1))
fc2_b=tf.Variable(tf.random_normal([OUTPUT_NODES],stddev=0.1,seed=1))

#定义第二个全连接层
y=tf.matmul(fc1,fc2_w)+fc2_b

#定义预测正确率和损失函数
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
loss=tf.reduce_mean((y-y_)*(y-y_))

#选择tf.train.AdamOptimizer优化器，优化对象是最小化损失函数
train_step=tf.train.AdamOptimizer(RATE).minimize(loss)

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

dataset_size=np.shape(X_tr)[0]

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(CYCLE_TIMES):
        start=(i*BATCH_SIZE)%dataset_size
        end=min(start+BATCH_SIZE,dataset_size)
        sess.run(train_step,feed_dict={x0:X_tr[start:end],y_:Y_tr[start:end]})
        if i%200==199 or i==0:
            loss1=sess.run(loss,feed_dict={x0:X_tr,y_:Y_tr})#训练集上的损失函数
            accuracy1=sess.run(accuracy,feed_dict={x0:X_tr,y_:Y_tr})#训练集上的分类正确率
            loss2=sess.run(loss,feed_dict={x0:X_te,y_:Y_te})#测试集上的损失函数
            accuracy2=sess.run(accuracy,feed_dict={x0:X_te,y_:Y_te})#测试集上的分类正确率
            print("%6d  %.6f    %.6f    %.6f    %.6f"%(i+1,loss1,loss2,accuracy1,accuracy2))
    '''
    for i in range(50):
        data=scio.loadmat('clyc'+str(i+1)+'_2.mat')
        X=data['y2']
        CNN_results=sess.run(y,feed_dict={x0:X})
        scio.savemat('CNN_results'+str(i+1)+'.mat', {'CNN_results': CNN_results}) 
    '''
