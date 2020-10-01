import json
import os
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

sessionData = {}
images = []

def findSessionIndex(sessionName, data):
  for i in range(len(data)):
    if ('sessionName' in data[i] and data[i]['sessionName'] == sessionName):
      return(i)
  else:
    return(-1)

def readPolyData(dataFile = 'data/polyData.json', verbose = False):
  if(verbose):
    print('Getting polydata from',dataFile)
  with open(dataFile, "r") as f:
    polyData = json.load(f)
    if(verbose):
      print('Successfully Read',dataFile)
    return(polyData)

def writePolyData(sessionData, dataFile = 'data/polyData.json', verbose = False):
  if(verbose):
    print('Loading polydata file from',dataFile)
  with open(dataFile, "r") as f:
    polyData = json.load(f)
    if(verbose):
      print('Successfully Read',dataFile)
    i = findSessionIndex(sessionData['sessionName'], polyData['data'])
    if(i == -1):
      i = len(polyData['data'])
      polyData['data'].append({})
    polyData['data'][i] = sessionData

  with open(dataFile, 'w') as f:
    json.dump(polyData, f)
    return(True)

def readImages(dir = 'data/brain' ,verbose = False):
  if(verbose):
    print('Reading Images from',dir)
  allowedImgFormats = ['png','jpg']
  imgs = [dir+'/'+img for img in os.listdir(dir) if img.split('.')[-1] in allowedImgFormats]
  imgs = sorted(imgs, key = lambda x: int(x.split('/')[-1].split('.')[0]))
  if (verbose):
    print('Read %i Images'%(len(imgs)))

  return(imgs)

# FUNCTION THAT PRINTS AN IMAGE WITH THE SEGMENTATION GIVEN BY contour VECTOR 
# THSI WAS ALSO USED IN MIDRERM SUBMISSION
def segmentedImg(img, contour):
  try:
    img = cv2.imread(img)
  except:
    print('Error while reading image, File may be corrupted.')

  resImg = img[:]
  for x in contour:
    i = x[0]
    j = x[1]
    points = [
      [i-1, j-1],
      [i-1, j],
      [i-1, j+1],
      [i, j-1],
      [i, j],
      [i, j+1],
      [i+1, j-1],
      [i+1, j],
      [i+1, j+1]
    ]
    for pt in points:
      if(pt[0] < img.shape[0] and pt[1] < img.shape[1] and pt[0] > 0 and pt[1] > 0):
        resImg[pt[0]][pt[1]] = [0, 0, 255]
  return(resImg)

def initializeContour(imgPath, pointDistance = 10 ,verbose = False):
  try:
    img = cv2.imread(imgPath)
  except:
    if(verbose):
      print('Error while reading image, File may be corrupted.')
    return()
  m = img.shape[0]
  n = img.shape[1]

  if(verbose):
    print('Initializing contours for image with dimensions %ix%i'%(m,n))

  contour = []

  #No longer required
  padding = 0
  
  left = [[x, pointDistance] for x in range(m) if (x%pointDistance == 0 and x > padding and m-padding-x > pointDistance)][::-1]
  right  = [[x, n - pointDistance] for x in range(m) if (x%pointDistance == 0 and x > padding and m-padding-x > pointDistance)]
  top = [[pointDistance, x] for x in range(n) if (x%pointDistance == 0 and x > padding and n-padding-x > pointDistance)]
  bottom = [[m - pointDistance, x] for x in range(n) if (x%pointDistance == 0 and x > padding and n-padding-x > pointDistance)][::-1]
  
  for x in top + right + bottom + left:
    if (x not in contour):
      contour.append(x)

  if(verbose):
    print('Generated Initial Contour with %i points'%(len(contour)))
    print(contour)

  return(contour)

def calc_avg_dist(contour_avg):
  total_distance = 0
  for i in range(len(contour_avg)):
    curr_element = contour_avg[i]
    prev_element = contour_avg[i-1]
    total_distance += ((curr_element[0] - prev_element[0]) ** 2 + (curr_element[1] - prev_element[1]) ** 2) ** 0.5
  avg_distance = total_distance/len(contour_avg)
  return (avg_distance)

def calc_e_cont(contour_e_cont, curr_element, prev_element ):
    d = calc_avg_dist(contour_e_cont)
    e_cont = d - ((curr_element[0] - prev_element[0])**2 + (curr_element[1] - prev_element[1])**2)
    return(e_cont)

def calc_e_curve(curr_element,prev_element,next_element):
  e_curve_x = (prev_element[0] - 2*curr_element[0] + next_element[0]) ** 2
  e_curve = e_curve_x + (prev_element[1] - 2*curr_element[1] + next_element[1])**2
  return(e_curve)

def calc_image_gradient(img):
  gradient_mag = np.zeros_like(img)
  for i in range(1, img.shape[0]):
    for j in range(1, img.shape[1]):
      gradient_mag[i][j] = int((img[i][j]-img[i][j-1] )**2 + (img[i][j] - img[i-1][j])**2)
  return gradient_mag

# TRIED USING MAXFILTER ON THE GRADIENT MAGNITUDE TO GET FINER RESULT
def maxFilter(img, kSize = 3):
  resImg = np.zeros_like(img)
  m = img.shape[0]
  n = img.shape[1]
  kDist = int(kSize/2)
  for i in range(m):
    for j in range(n):
      if(i < kDist or j < kDist or i >= m-kDist or j >= n-kDist):
        resImg[i][j] = 0
      else:
        neighborhood = []
        for p in range(-kDist,kDist+1):
          for q in range(-kDist, kDist+1):
            neighborhood.append([i+p,j+q])

        resImg[i][j] = max([img[x,y] for x,y in neighborhood])
  return(resImg)

### SEGMENTATION ALGORITHM (RUNS ONCE FOR EACH ITERATION OF THE SEGMENTATION)
def segment(segItr, contour, image_grad, imgName = 'testImg', verbose = False, maxIters = 200, alpha = 0.01, beta = 0.05, gamma = 20):
  flag = False
  if(segItr < maxIters):
    change = 0
    segItr += 1
    for point in contour:
      point_index = contour.index(point)
      x = point[0]
      y = point[1]
      neighbourhood =[[x-1,y-1],[x-1,y],[x-1,y+1], [x,y-1],[x,y],[x,y+1], [x+1,y-1],[x+1,y],[x+1,y+1]]
      e_min = None
      e_min_pt = None
      for index in range(len(neighbourhood)):
        curr_pt = neighbourhood[index]
        prev_point = contour[point_index-1]
        if point_index < len(contour)-1:
          next_point = contour[point_index +1]
        elif point_index == len(contour) -1:
          next_point = contour[0]
        temp_contour = copy.deepcopy(contour)
        temp_contour[point_index] = curr_pt
        e_point = alpha*calc_e_cont(contour_e_cont=temp_contour, curr_element =curr_pt, prev_element=prev_point)
        e_point += beta*calc_e_curve(curr_element=curr_pt, next_element=next_point, prev_element=prev_point)
        e_point += gamma*image_grad[curr_pt[0]][curr_pt[1]]
        if e_min is None or e_point < e_min:
          e_min = e_point
          e_min_pt = curr_pt
      if e_min_pt != point:
        curr_pt_index = contour.index(point)
        contour[curr_pt_index] = e_min_pt
        change += 1
    if (change == 0):
      flag = True

  else:
    if(verbose):
      print('Segmentation Complete')
    flag = True

  return(contour, flag)

def testSegmentation(imgPath, verbose = False):
  img = []

  try:
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
  except:
    if(verbose):
      print('Error while reading image, File may be corrupted.')
    return()
  if(verbose):
    print('Image Read with shape:',len(img),'x',len(img[0]))
  
  Alpha = 1
  Beta = 1
  Gamma = 1
  pointDist = 20
  maxItr = 200

  contour = initializeContour(imgPath, pointDistance = pointDist, verbose = False)
  gMag = img = calc_image_gradient(img)

  if(verbose):
    plt.imshow(gMag, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.show()

  if(verbose):
    print('Starting Segmentation')
  flag = False

  i = 0
  while(not flag):
    contour, flag = segment(segItr = i, image_grad = gMag, imgName = imgPath, contour = contour, maxIters=maxItr, alpha = Alpha, beta = Beta, gamma = Gamma, verbose = verbose)
    if(verbose and (i%(maxItr/5)) == 0):
      fig = plt.figure(figsize = (7,7))
      plt.imshow(segmentedImg(imgPath, contour), cmap='gray')
      plt.title('ITR: %i'%(i))
      plt.show()
    i += 1

  fig = plt.figure(figsize = (7,7))
  plt.imshow(segmentedImg(imgPath, contour), cmap='gray')
  plt.title('FINAL')
  plt.show()

#testSegmentation('test/testimg.png', verbose=True)