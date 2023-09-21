import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./anh.jpg")
img=cv2.resize(img,(400,400))
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def conv2d(input,kenelSize):
    height,width = input.shape
    kenel=np.random.randn(kenelSize,kenelSize)
    result=np.zeros((height-kenelSize+1,width-kenelSize+1))
    def getROI():
        for row in range(height-kenelSize+1):
            for col in range(width-kenelSize+1):
                roi = input[row:row+kenelSize,col:col+kenelSize]
                yield roi , row,col
    def operate():
        for roi,row,col in getROI():
            result[row,col]=np.sum(roi*kenel)
    operate()
    return result

# imgaffer= conv2d(img,3)
# while True :

#     cv2.imshow("image", imgaffer)
#     if cv2.waitKey(1) == ord('q'):
#         break
class Conv2d:
    def __init__(self,input,numOfKenel=3,kenelSize=3,padding=0,stride=1) -> None:
        self.input=np.pad(input,((padding,padding),(padding,padding)),'constant')
        self.stride=stride
        self.kenel =np.random.randn(numOfKenel,kenelSize,kenelSize)
        
        self.results=np.zeros((int((self.input.shape[0]-self.kenel.shape[1])/self.stride)+1,
                               int((self.input.shape[1]-self.kenel.shape[2])/self.stride)+1,
                               self.kenel.shape[0]))

    def getROI(self):
        for row in range(int((self.input.shape[0]-self.kenel.shape[1])/self.stride)+1):
            for col in range(int((self.input.shape[1]-self.kenel.shape[2])/self.stride)+1): 
                roi = self.input[row*self.stride:row*self.stride+self.kenel.shape[1],
                                 col*self.stride:col*self.stride+self.kenel.shape[2]]   
                yield row,col,roi
    def operate(self):
        for layer in range(self.kenel.shape[0]):
            for row,col ,roi in self.getROI(): 
                self.results[row,col,layer] = np.sum(roi*self.kenel[layer])  
        return self.results
class Relu:
    def __init__(self,input) -> None:
        self.input=input
        self.result=np.zeros((self.input.shape[0],self.input.shape[1],self.input.shape[2]))
    
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row,col,layer]= 0 if self.input[row,col,layer] < 0 else self.input[row,col,layer]
        return self.result

class LeakyRelu:
    def __init__(self,input) -> None:
        self.input=input
        self.result=np.zeros((self.input.shape[0],self.input.shape[1],self.input.shape[2]))
    
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row,col,layer]= 0.1*self.input[row,col,layer] if self.input[row,col,layer] < 0 else self.input[row,col,layer]
        return self.result
class MaxPooling:
    def __init__(self,input,poolingSize) -> None:
        self.input=input
        self.poolingSize=poolingSize
        self.result=np.zeros((int(self.input.shape[0]/self.poolingSize),
                              int(self.input.shape[1]/self.poolingSize),
                              self.input.shape[2])) 
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.poolingSize)):
                for col in range(int(self.input.shape[1]/self.poolingSize)):
                    self.result[row,col,layer]=np.max(self.input[row*self.poolingSize:row*self.poolingSize+self.poolingSize,
                                                          col*self.poolingSize:col*self.poolingSize+self.poolingSize,
                                                          layer])
        return self.result
class Solfmax:
    def __init__(self,input,nodes) -> None:
        self.input=input
        self.nodes=nodes
        # y =w0+w(i)*x
        self.flatten=self.input.flatten()
        self.weight = np.random.randn(self.flatten.shape[0])/self.flatten.shape[0]
        self.bias=np.random.randn(nodes)
    def operate(self):
        totals=np.dot(self.flatten,self.weight)+self.bias
        exp=np.exp(totals)
        return exp/np.sum(exp)
conv2d = Conv2d(img,numOfKenel=8,kenelSize=3,padding=10,stride=1)

img_gray_conv2d=conv2d.operate()
img_gray_conv2d_relu=LeakyRelu(img_gray_conv2d).operate()
img_gray_conv2d_maxpooling=MaxPooling(img_gray_conv2d_relu,3).operate()
solfmax = Solfmax(img_gray_conv2d_maxpooling,10)
print(solfmax.operate())
# for i in range(8):
#     plt.subplot(2,4,i+1)
#     plt.imshow(img_gray_conv2d[ :, :, i],cmap='gray')
#     plt.axis('off')
# plt.savefig('img_gray_conv2d_leakyrelu_maxpooling.jpg')
# plt.show()

    