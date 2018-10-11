# simply_q-learning

# ref http://mnemstudio.org/path-finding-q-learning-tutorial.htm

![ql](http://mnemstudio.org/ai/path/images/map2a.gif)

```
import numpy as np
```

```
# define two matrix for reward and Q
# rewardMatrix contains the reward and path
rewardMatrix=-np.ones((6,6))
qMatrix=np.zeros((6,6))((6,6))
```


```
# let's setup rewardMatrix 
rewardMatrix[0,4]=0
rewardMatrix[1,3]=0
rewardMatrix[1,5]=100
rewardMatrix[2,3]=0
rewardMatrix[3,1]=0
rewardMatrix[3,2]=0
rewardMatrix[3,4]=0
rewardMatrix[4,0]=0
rewardMatrix[4,3]=0
rewardMatrix[4,5]=100
rewardMatrix[5,1]=0
rewardMatrix[5,4]=0
rewardMatrix[5,5]=100
print(rewardMatrix)
print(qMatrix)
'''
[[ -1.  -1.  -1.  -1.   0.  -1.]
 [ -1.  -1.  -1.   0.  -1. 100.]
 [ -1.  -1.  -1.   0.  -1.  -1.]
 [ -1.   0.   0.  -1.   0.  -1.]
 [  0.  -1.  -1.   0.  -1. 100.]
 [ -1.   0.  -1.  -1.   0. 100.]]
[[0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0.]]
'''
```
# Training of Q-table
```
nEpisode=200
r=0.1
endState=5

for i in range(nEpisode):    
    startPoint=np.random.randint(6)
    print(i,'startPoint',startPoint)
    canMove=True
    while(canMove):
        actionPool=np.where(rewardMatrix[startPoint,:]>=0)[0]  # find available path
        newAction=np.random.choice(list(actionPool),1)[0]      # random pick 
        #Q(s,a)=R(s,a)+r*max{Q(~s,~a)}
        thisReward=rewardMatrix[startPoint,newAction]    #R(s,a)
        futureActionPool=np.where(rewardMatrix[newAction,:]>=0)[0]
        qMatrix[startPoint,newAction]=thisReward+r*np.max(qMatrix[newAction,futureActionPool]) #R(s,a)+r*max{Q(~s,~a)}
        print(actionPool,'startPoint',startPoint,newAction,'thisReward',thisReward,furtureActionPool,qMatrix[newAction,futureActionPool],'Matrix %d %d='%(startPoint,newAction),qMatrix[startPoint,newAction])
        if newAction==endState:
            canMove=False
            print(qMatrix)
        else:
            startPoint=newAction
```


# Testing of Q-learning

```
# we need only qmatrix here
# each time pick an action with the max Q value

testPoint=0
canMove=True
while(canMove):
    
    actionPool=np.argmax(qMatrix[testPoint,:])
    print(qMatrix[testPoint,:],actionPool)
    if actionPool==endState:
        canMove=False
    else:
        testPoint=actionPool
        
 '''
 [0.          0.          0.          0.         11.11111111  0.      ]   4
 [1.11111111   0.           0.           1.11111111   0.  111.11111111]   5
 '''

```
