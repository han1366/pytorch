from Alex.AlexNet import AlexNet
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable


class AlexTest():
    def __init__(self,n_output):
        self.classes = {0:'plane', 1:'car', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        self.my_model = AlexNet(n_output)
        self.my_model.load_state_dict(torch.load('AlexNetparams.pkl'))


    def test(self):
        img = loadImg()
        data = img.getdata()
        data = np.array(data, dtype='float') / 255.0
        new_data = np.reshape(data * 255.0, (3,224, 224))  # 矩阵
        new_data = new_data[np.newaxis, :]
        print(new_data.shape)
        new_data = Variable(torch.from_numpy(new_data).float())
        pred = self.my_model(new_data)  # Variable型的，返回值必须转化成list型
        print(self.classes.get(np.argmax(pred[0].detach().numpy())))

def loadImg():
    rimg = Image.open('/home/hanlei/桌面/dog.png').convert('RGB')           #将四通道的RGBA图像专成RGB图像
    rimg = rimg.resize((224,224))
    return rimg

if __name__ == '__main__':
    output = 10
    Alex = AlexTest(output)
    Alex.test()