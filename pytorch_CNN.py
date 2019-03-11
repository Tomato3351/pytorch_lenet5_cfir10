# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:40:18 2019

@author: TOMATO
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2)
#        print('x_size =',x.size())
        x=x.view(-1,self.num_flat_features(x))
#        print('flat_size =',x.size())
        #先进行一次forward，计算第一个全连接层参数个数
#        x=F.relu(nn.Linear(x.size()[1],120)(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
        
    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features *=s
        return num_features

def transpose2np(img):
    #将CIFAR10数据集中的图片(3,32,32),转换为np.array,同时size变为(32,32,3)
    img=img/2+0.5
    npimg=img.numpy()
    transposed=np.transpose(npimg,(1,2,0))
    return transposed

if __name__=="__main__":
#    img_tensor=torch.randn(1,3,32,32)    
#    net.forward(img_tensor)
    transform=transforms.Compose(
        [transforms.ToTensor(),
         #归一化到[-1,1]
         transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
   
    trainset=torchvision.datasets.CIFAR10(root='./torchvision_data',train=True,
                                          download=False,transform=transform)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,
                                            shuffle=True,num_workers=2)#多线程
    testset=torchvision.datasets.CIFAR10(root='./torchvision_data',train=False,
                                         download=False,transform=transform)
    testloader=torch.utils.data.DataLoader(testset,batch_size=4,
                                           shuffle=False,num_workers=2)
    classes=np.array(['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse',
             'ship', 'truck'])
    
    show_batch_num=6#要显示的batch数
    plt.figure("train images",figsize=(5,8))
    #显示部分训练图像
    for i in range(show_batch_num*4):
        image,label=trainset[i]
        plt.subplot(show_batch_num,4,i+1)
        plt.title(classes[label])
        image=transpose2np(image)
        plt.imshow(image)#cmap='gray'
        plt.axis('off')

#    device=torch.device("cpu")
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
#    nvidia-smi
    print(device)
    net=Net().to(device)
#
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    start_training=time.perf_counter()     
    for epoch in range(1):#训练两次
        print('Start training epoch',epoch+1)
        running_loss=0.0
        for i,data in enumerate(trainloader,0):           
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
#            #梯度置0
            optimizer.zero_grad()      
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
#            #打印状态信息
            running_loss+=loss.item()
            if i%1000==999:#打印
                print('[%d,%5d] loss: %.3f'%(
                      epoch+1,i+1,running_loss/2000))
#                print('epoch:',epoch+1,'batches:',i+1,
#                      'running_loss:',running_loss/2000)
                running_loss=0.0
    end_training=time.perf_counter() #计时结束
    print('Training time:%d s'%(end_training-start_training))  
    print('Training Finished')
#
    plt.figure("test images",figsize=(5,8))
    #显示部分测试图像
    test_start_num=0
    print('Testing...')
    for i in range(test_start_num,test_start_num+show_batch_num*4):
        image,label=testset[i]
        image_exdim=np.expand_dims(image,axis=0)
        image_exdim_tensor = torch.from_numpy(image_exdim).to(device)
        output=net(image_exdim_tensor)
        _,predicted=torch.max(output,1)
        plt.subplot(show_batch_num,4,i+1)
        plt.title(classes[label]+'-'+classes[predicted])
        image=transpose2np(image)
        plt.imshow(image)#cmap='gray'
        plt.axis('off')
#
    correct=0
    total=0
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            outputs=net(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    print('Accuracy of the network on the testimages:%d %%'%
          (100*correct/total))
#        
    class_correct=list(0. for i in range(10))
    class_total=list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images,labels=data
            images,labels=images.to(device),labels.to(device)
            outputs=net(images)
            _,predicted=torch.max(outputs.data,1)
            c=(predicted==labels)
            for i in range(4):
                label=labels[i]
                class_correct[label]+=c[i].item()
                class_total[label]+=1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %(
                classes[i],100*class_correct[i]/class_total[i]))

#    torch.cuda.empty_cache()
   
#    outputs=net(testset[0:12,:,:,:])
    
#    img_tensor=torch.randn(1,1,45,45)
#    print('input=\n',img_tensor)
#    print('input size=\n',img_tensor.size())
##    print(img_tensor[:,:,:,0])
##    cv2.namedWindow('img',0)
##    cv2.imshow('img',img[0,:,:,:])
##    cv2.waitKey(0)
#    net=Net()
#    print('net=',net)    
#    params=list(net.parameters())
#    print('params_len =',len(params))
#    for i in range(len(params)):
#        print(i,'param size=',params[i].size()) 
#    out=net(img_tensor)
#    print('out =',out)
#    net.zero_grad()
#    out.backward(torch.randn(1,10))
#    
#    
#    output=net(img_tensor)
#    target=torch.randn(10)
#    target=target.view(1,-1)
#    criterion=nn.MSELoss()
#    loss=criterion(output,target)
#
#    print('conv1.bias.grad before backward')
#    print(net.conv1.bias.grad)
#    
#    loss.backward()
#    
#    print('conv1.bias.grad after backward')
#    print(net.conv1.bias.grad)    
#    
#    
#    optimizer=optim.SGD(net.parameters(),lr=0.01)
#    
#    #in training loop:
#    optimizer.zero_grad()
#    output=net(img_tensor)
#    loss=criterion(output,target)
#    loss.backward()
#    optimizer.step()
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    










