#coding:utf8
import shortuuid
import torch
import torch.nn as nn
from torch.autograd import Variable
try:
    from .datasets import CaptchaData,CaptchaDataOne
    from .CONFIG import LettersInt
except:
    from datasets import CaptchaData,CaptchaDataOne
    from CONFIG import LettersInt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import time
from PIL import Image
import os, requests
import base64
from io import BytesIO


class CNN(nn.Module):
    def __init__(self, range_len=LettersInt.range_len, pic_name_len=LettersInt.PIC_NAME_LEN):
        super(CNN, self).__init__()
        self.range_len = range_len
        self.pic_name_len = pic_name_len

        IMG_WIDTH = LettersInt.IMG_WIDTH
        IMG_HEIGHT = LettersInt.IMG_HEIGHT

        IMG = 3  # RGB
        input1 = 3
        out1 = 16
        maxpool2d1 = 2
        IMG_HEIGHT = int(IMG_HEIGHT / maxpool2d1)
        IMG_WIDTH = int(IMG_WIDTH / maxpool2d1)
        input2 = 16
        out2 = 64
        maxpool2d2 = 2
        IMG_HEIGHT = int(IMG_HEIGHT / maxpool2d2)
        IMG_WIDTH = int(IMG_WIDTH / maxpool2d2)
        input3 = 64
        out3 = 512
        maxpool2d3 = 2
        IMG_HEIGHT = int(IMG_HEIGHT / maxpool2d3)
        IMG_WIDTH = int(IMG_WIDTH / maxpool2d3)

        self.linear = out3 * IMG_WIDTH * IMG_HEIGHT
        self.conv = nn.Sequential(
            # batch*3*180*100 # IMG_WIDTH,IMG_HEIGHT=100,30
            nn.Conv2d(input1, out1, IMG, padding=(1, 1)),
            # 参数分别对应着输入的通道数3，输出通道数16,卷积核大小为3（长宽都为3）adding为（1， 1）可以保证输入输出的长宽不变
            nn.MaxPool2d(maxpool2d1, maxpool2d1),
            nn.BatchNorm2d(out1),
            nn.ReLU(),
            # batch*16*90*50
            nn.Conv2d(input2, out2, IMG, padding=(1, 1)),
            nn.MaxPool2d(maxpool2d2, maxpool2d2),
            nn.BatchNorm2d(out2),
            nn.ReLU(),
            # batch*64*45*25
            nn.Conv2d(input3, out3, IMG, padding=(1, 1)),
            nn.MaxPool2d(maxpool2d3, maxpool2d3),
            nn.BatchNorm2d(out3),
            nn.ReLU(),
            # batch*512*22*12
            # nn.Conv2d(input4, out4, IMG, padding=(1, 1)),
            # nn.MaxPool2d(maxpool2d4, maxpool2d4),
            # nn.BatchNorm2d(out4),
            # nn.ReLU(),
            # batch*512*11*6
        )
        self.fc = nn.Linear(self.linear, self.range_len * self.pic_name_len)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.linear)
        x = self.fc(x)
        return x

class CrackLettesInt4:

    def __init__(self):
        self.batch_size = 1
        self.base_lr = 0.001
        self.max_epoch = 200
        self.model_path = LettersInt.model_path
        self.restor = False
        self.range_len = LettersInt.range_len

    def calculat_acc(self,output, target):
        output, target = output.view(-1, LettersInt.range_len), target.view(-1, LettersInt.range_len)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        output, target = output.view(-1, LettersInt.PIC_NAME_LEN), target.view(-1, LettersInt.PIC_NAME_LEN)
        correct_list = []
        for i, j in zip(target, output):
            if torch.equal(i, j):
                correct_list.append(1)
            else:
                correct_list.append(0)
        acc = sum(correct_list) / len(correct_list)
        return acc

    def train(self,train_folder):
        transforms = Compose([ToTensor()])
        train_dataset = CaptchaData(train_folder, transform=transforms)
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True)
        cnn = CNN()
        if torch.cuda.is_available():
            cnn.cuda()
        # if self.restor:
        #     cnn.load_state_dict(torch.load(LettersInt.model_path))
        #        freezing_layers = list(cnn.named_parameters())[:10]
        #        for param in freezing_layers:
        #            param[1].requires_grad = False
        #            print('freezing layer:', param[0])

        optimizer = torch.optim.Adam(cnn.parameters(), lr=self.base_lr)
        criterion = nn.MultiLabelSoftMarginLoss()

        for epoch in range(self.max_epoch):
            start_ = time.time()
            loss_history = []
            acc_history = []
            cnn.train()
            for img, target in train_data_loader:
                img = Variable(img)
                target = Variable(target)
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                output = cnn(img)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = self.calculat_acc(output, target)
                acc_history.append(acc)
                loss_history.append(loss)
            print('train_loss: {:.4}|train_acc: {:.4}'.format(
                torch.mean(torch.Tensor(loss_history)),
                torch.mean(torch.Tensor(acc_history)),
            ))
            cnn.eval()
            print('epoch: {}|time: {:.4f}'.format(epoch, time.time() - start_))
            torch.save(cnn.state_dict(), LettersInt.model_path)


    def predict_one(self,image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        transforms = Compose([ToTensor()])
        img = transforms(img)
        cnn = CNN()
        if torch.cuda.is_available():
            cnn = cnn.cuda()
        cnn.eval()
        if torch.cuda.is_available():
            cnn.load_state_dict(torch.load(LettersInt.model_path))
        else:
            cnn.load_state_dict(torch.load(LettersInt.model_path,map_location=torch.device('cpu')))
        try:
            img = img.view(1, 3, LettersInt.IMG_HEIGHT, LettersInt.IMG_WIDTH).cuda()
        except:
            img = img.view(1,3,LettersInt.IMG_HEIGHT, LettersInt.IMG_WIDTH)
        output = cnn(img)
        output = output.view(-1, self.range_len)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, LettersInt.PIC_NAME_LEN)[0]
        pred = ''.join([LettersInt.content_range[i] for i in output.cpu().numpy()])
        print('result',pred)
        os.system('mv %s ./test/%s.jpg'%(image_path, pred))
        return pred

    @classmethod
    def down_img(cls,url,c):
        for i in range(c):
            res = requests.get(url).content
            g = open('./data/test/%s.jpg' % int(time.time()*1000),'wb')
            g.write(res)
            g.close()
            print(i)

    @classmethod
    def check_name(cls):
        pp = os.path.dirname(os.path.abspath(__file__))
        for d in os.listdir('./data/train'):
            if len(d) != 8:
                print(d)
                os.remove(pp+'\\data\\train\\'+d)

    def check_all(self,folder):
        for d in os.listdir(folder):
            f = './data/test/%s' % d
            self.predict_one(f)


def getCaptcha():
    url = 'https://ebanking.baca-bank.vn/IBSRetail/servlet/ImageServlet'
    headers = {}
    response = requests.get(url, headers=headers, timeout=15)
    return base64.b64encode(response.content).decode('utf-8')


def img_base4_save(img_base64, savePath):
    '''
    Base64图片保存本地
    '''
    # if not img_base64.startswith('data:image/png;base64,'):
    #     return

    # 去除前缀信息（如'data:image/png;base64,'）
    if img_base64.startswith('data:'):
        _, data = img_base64.split(',', maxsplit=1)
    else:
        data = img_base64

    # 解码Base64数据
    decoded_data = base64.b64decode(data)
    # 创建BytesIO对象来读取二进制数据
    buffered_io = BytesIO()
    buffered_io.write(decoded_data)
    buffered_io.seek(0)
    # 打开图像文件
    img = Image.open(buffered_io)
    # 指定保存路径及名称
    # 保存图像到本地
    img.save(savePath)
    return True


def ttt():
    from PIL import Image
    # 打开图片
    img = Image.open('1709659296.jpg')

    # 获取图片的宽度和高度
    width, height = img.size

    # 计算每个子图的尺寸
    sub_width = width // 4

    # 分割图片
    for j in range(4):
        if j == 4:
            box = (j * sub_width, 0, (j + 1) * sub_width+3, height)
        elif j == 5:
            box = (j * sub_width+3, 0, (j + 1) * sub_width, height)
        else:
            box = (j * sub_width, 0, (j + 1) * sub_width, height)
        sub_img = img.crop(box)
        sub_img.save(f'./t/sub_img_{j}.jpg')



def getImg():
    for i in range(20):
        print(i)
        url = 'https://online.vietbank.com.vn/ibk/vn/login/capcha.jsp'
        headers = {}
        response = requests.get(url, headers=headers, timeout=20)
        with open('./data/test/' + shortuuid.uuid() + '.jpg', 'wb') as wf:
            wf.write(response.content)



if __name__ == "__main__":
    # url = 'https://erp.hupun.com/service/captcha/generate?time=1638451991727&refresh=true'
    # CrackLettesInt4.down_img(url,100)
    CrackLettesInt4().check_all('./data/test')
    # CrackLettesInt4().train('./data/train')
    # dd = getCaptcha()
    # getImg()

    # for d in os.listdir('./data/train'):
    #     print('d:', d)
    #     if d.startswith('fff'):
    #         os.remove('./data/train/'+d)

