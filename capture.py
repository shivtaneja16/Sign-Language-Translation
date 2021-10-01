import numpy as np
import operator
import cv2
import sys, os
from keras.models import load_model
from keras.models import model_from_json
import json
from PIL import Image
import pygame

pygame.init()
screen = pygame.display.set_mode((900,900),pygame.RESIZABLE)

CLIP_X1 = 160
CLIP_Y1 = 140
CLIP_X2 = 400
CLIP_Y2 = 360


with open('model.json','r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(json.dumps(model_json))
loaded_model.load_weights('epoch5_model.h5')

#loaded_model = load_model('first.h5')

cap = cv2.VideoCapture(0)

while True:
    _, FrameImage = cap.read()
    FrameImage = cv2.flip(FrameImage, 1)
    cv2.imshow("", FrameImage)
    cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,1)

    ROI = FrameImage[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2]
    ROI = cv2.resize(ROI, (200, 200)) 
    #ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    #ROI= cv2.add(ROI,np.array([40.0]))
    _, output = cv2.threshold(ROI, 200, 255, cv2.THRESH_BINARY) # adjust brightness
    
    '''
    SHOWROI = cv2.resize(ROI, (256, 256)) 
    _, output2 = cv2.threshold(SHOWROI, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", output2)
    '''
    result = loaded_model.predict(np.reshape(ROI, [-1, 200, 200, 3]))
    predict =   { 'A':    result[0][0],
                  'B':    result[0][1],    
                  'C':    result[0][2],
                  'D':    result[0][3],
                  'E':    result[0][4],
                  'F':    result[0][5],
                  'G':    result[0][6],
                  'H':    result[0][7],    
                  'I':    result[0][8],
                  'J':    result[0][9],
                  'K':    result[0][10],
                  'L':    result[0][11],
                  'M':    result[0][12],
                  'N':    result[0][13],
                  'O':    result[0][14],    
                  'P':    result[0][15],
                  'Q':    result[0][16],
                  'R':    result[0][17],
                  'S':    result[0][18],
                  'T':    result[0][19],
                  'U':    result[0][20],
                  'V':    result[0][21],
                  'W':    result[0][22],    
                  'X':    result[0][23],
                  'Y':    result[0][24],
                  'Z':    result[0][25],
                  'del':    result[0][26],
                  'nosign':    result[0][27],
                  'space':    result[0][28],
                  }
    
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    
    if(predict[0][1] >= 1.0):
        predict_img  = pygame.image.load(os.getcwd() + '/dataset/images/' + predict[0][0] + '.png')
    else:
        predict_img  = pygame.image.load(os.getcwd() + '/dataset/images/nosign.png')
    predict_img = pygame.transform.scale(predict_img, (900, 900))
    screen.blit(predict_img, (0,0))
    pygame.display.flip()
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('q'): # esc key
        break
            
pygame.quit()
cap.release()
cv2.destroyAllWindows()
