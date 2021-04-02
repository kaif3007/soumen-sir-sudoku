import cv2
from PIL import Image
import numpy as np
from skimage.segmentation import clear_border
import imutils
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sudoku import Sudoku

model=load_model('second_trail.h5')

def extract_digit(cell, debug=True):

  thresh = cv2.threshold(cell, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  
  thresh=clear_border(thresh)

  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts=imutils.grab_contours(cnts)

  if len(cnts)==0:
    return None
  
  c = max(cnts, key=cv2.contourArea)
  mask = np.zeros(thresh.shape, dtype="uint8")
  cv2.drawContours(mask, [c], -1, 255, -1)
  
  digit = cv2.bitwise_and(thresh, thresh, mask=mask)
  return digit

def my_func(warped):

  board = np.zeros((9, 9), dtype="int")

  stepX = warped.shape[1] // 9
  stepY = warped.shape[0] // 9
  
  cellLocs = []

  for y in range(0, 9):
    row = []
    for x in range(0, 9):
      startX = x * stepX
      startY = y * stepY
      endX = (x + 1) * stepX
      endY = (y + 1) * stepY
    
      row.append((startX, startY, endX, endY))
      cell = warped[startY:endY, startX:endX]
      cell = cv2.resize(cell,(28, 28))
      digit=extract_digit(cell)

      if digit is not None:
     
        digit = digit.astype("float") / 255.0
        digit = img_to_array(digit)
        digit = np.expand_dims(digit, axis=0)
        
        pred = model.predict(digit).argmax(axis=1)[0]
        board[y, x] = pred
	
    cellLocs.append(row)

  puzzle = Sudoku(3, 3, board=board.tolist())
  puzzle.show()

  solution = puzzle.solve()
  solution.show_full()
  print(extract_digit(cv2.resize(warped[cellLocs[1][4][1]:cellLocs[1][4][3],cellLocs[1][4][0]:cellLocs[1][4][2]],(28,28)),1))

img=cv2.imread('perspective.jpg',0)
my_func(img)
