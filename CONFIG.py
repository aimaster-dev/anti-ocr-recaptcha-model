#coding:utf8
import os
import string


source = list('123456789ABDEFGHJMNRTYabdefghjmnqrty')

class LettersInt:
    content_range = ''.join(source)
    range_len = len(content_range)
    IMG_WIDTH = 200
    IMG_HEIGHT = 35
    PIC_NAME_LEN = 6
    model_path = os.path.dirname(os.path.realpath(__file__)) + '/model.pth'
