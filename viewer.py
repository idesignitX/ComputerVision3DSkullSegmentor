import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import segmentor as seg
import copy
import cv2

import time ##TEMP, REMOVE !!

#IMPORTS
  # pygame
  # pyOpengl
  # os [Just to hide the annoying pygame print statement on import]
  # json
  # openCV (cv2)
  # copy [For deepcopy]
  # numpy

modelView = []
sliceView = []

sessionData = {'contourList': []}

settings = {
  'segItr' : 0,
  'imgItr' : 0,
  'ySeparation' : 0.05,
  'Faces' : False,
  'Lines' : True,
  'Points' : True,
  'Slices' : True,
  'separationFactor' : 1,
  'taskComplete' : False
}

def sqDist(li1, li2):
  return(sum([(a-b)**2 for a,b in zip(li1,li2)]))

# SLICES NEED TO HAVE SAME NUMBER OF VERTICES
def genLayer(contour1, contour2, layer = 0, ySeparation = 0.1, verbose = False):
  i = 0
  j = contour2.index(min(contour2, key = lambda x: sqDist(contour1[i], x)))%len(contour2)
  tris = []
  slicePoly = [[],[]]
  while i < len(contour1):
    h1 = layer * ySeparation
    h2 = (layer + 1) * ySeparation
    v1 = contour1[i]
    v1 = (v1[0], h1, v1[1])
    v2 = contour1[(i + 1)%len(contour1)]
    v2 = (v2[0], h1, v2[1])
    v3 = contour2[j]
    v3 = (v3[0], h2, v3[1])
    v4 = contour2[(j + 1)%len(contour2)]
    v4 = (v4[0], h2, v4[1])
    i += 1
    j = (j + 1)%len(contour2)
    tris.append((v1, v2, v3))
    tris.append((v3, v4, v2))
    slicePoly[0].extend([v1,v2])
    slicePoly[1].extend([v3,v4])
    if(i >= len(contour1)):
      break
  if(verbose):
    print('Stitching Complete with %i Triangles'%(len(tris)))
  return(tris,slicePoly)

def genModels(contourList, ySeparation = 0.1, verbose = False):
  n = len(contourList)
  i = 0
  while(True):
    if(verbose):
      print('Stitching layers: %i - %i'%(i, i+1))
    tris, slicePoly = genLayer(contourList[i], contourList[i+1], i, ySeparation, verbose)
    modelView.extend(tris)
    sliceView.extend(slicePoly)

    i += 1
    if(i+1 >= n):
      break

  if(verbose):
    print('Laters stitched')
    print('Model generated with %i triangles'%(len(modelView)))

def normalizeContourList(contourList, verbose = False):
  global sessionData
  if(verbose):
    print('Normalizing the contourlist')
    
  normalizedContourList = copy.deepcopy(contourList)
  img = cv2.imread(sessionData['imgs'][0])
  m = len(img)
  n = len(img[0])

  for c in normalizedContourList:
    for v in c:
      v[0] = ((v[0]/n)*2) - 1
      v[1] = ((v[1]/m)*2) - 1

  return(normalizedContourList)

def runLoop(verbose = False):
  global sessionData
  global settings

  pygame.init()
  display = (1366,768)
  pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
  gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

  glTranslatef(0,0,-5.5)
  glRotatef(35,1,0,0)
  glRotatef(90,0,1,0)
  glClearColor(0.6, 0.4, 0.8, 1.0)
  glEnable(GL_DEPTH_TEST)

  if(verbose):
    print('Starting Gameloop')
    print('Keyboard controls: ')
    print('W, S, A ,D, R, F     -    Camera movement')
    print('E, Q                 -    Camera Rotation')
    print('Z                    -    Reduce Slice separation')
    print('X                    -    Increase Slice separation')
    print('1                    -    Toggle Faces')
    print('2                    -    Toggle Lines')
    print('3                    -    Toggle Points')
    print('4                    -    Toggle Slices')
    print('ESC                  -    Exit')

  if(not settings['taskComplete']):
    initSegmentation(verbose = verbose)

  while(True):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
          pygame.quit()
          quit()

      #THIS IS WHEN ON KEY PRESS (EXECUTES ONCE)
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_1:
          settings['Faces'] = not settings['Faces']

        if event.key == pygame.K_2:
          settings['Lines'] = not settings['Lines']

        if event.key == pygame.K_3:
          settings['Points'] = not settings['Points']

        if event.key == pygame.K_4:
          settings['Slices'] = not settings['Slices']

        if event.key == pygame.K_ESCAPE:
          pygame.quit()
          quit()

    #THIS IS WHEN KEY IS HELD DOWN (EXECUTES CONTINOUSLY)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
      glTranslatef(0,0,0.03)

    if keys[pygame.K_s]:
      glTranslatef(0,0,-0.03)

    if keys[pygame.K_a]:
      glTranslatef(0.03,0,0)

    if keys[pygame.K_d]:
      glTranslatef(-0.03,0,0)

    if keys[pygame.K_r]:
      glTranslatef(0,-0.03,0)

    if keys[pygame.K_f]:
      glTranslatef(0,0.03,0)

    if keys[pygame.K_q]:
      glRotatef(0.8,0,1,0)  
    
    if keys[pygame.K_e]:
      glRotatef(0.8,0,-1,0)

    if keys[pygame.K_z]:
      settings['separationFactor'] -= 0.05
      if(settings['separationFactor'] <= 0):
        settings['separationFactor'] = 0.0001

    if keys[pygame.K_x]:
      if(settings['separationFactor'] <= 0.01):
        settings['separationFactor'] = 0.01
      settings['separationFactor'] += 0.05

    i = settings['imgItr']
    if(not settings['taskComplete']):
      
      img = sessionData['imgs'][i]
      #newContour, flag = seg.segment(segItr = settings['segItr'],img = img, contour = copy.deepcopy(sessionData['latestContour']), verbose = verbose)
      newContour, flag = seg.segment(segItr = settings['segItr'], 
                                    image_grad = sessionData['gMag'],                                                       \
                                    imgName = img,                                                                          \
                                    contour = copy.deepcopy(sessionData['latestContour']),                                  \
                                    maxIters = settings['maxItr'],                                                          \
                                    alpha = settings['Alpha'], beta = settings['Beta'], gamma = settings['Gamma'],          \
                                    verbose = verbose)
      if(flag):
        settings['imgItr'] += 1
        settings['segItr'] = 0
        initNewImage(verbose = verbose)
      else:
        if(i >= len(sessionData['rawContourList'])):
          sessionData['rawContourList'].append([])

        sessionData['rawContourList'][i] = newContour
        sessionData['latestContour'] = newContour
        sessionData['contourList'] = copy.deepcopy(sessionData['rawContourList'])
        updateView()
        settings['segItr'] += 1
      

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
    if(settings['Slices']):
      glColor3f(1.0,0.6,0.8)
      for layer in sliceView:
        glBegin(GL_POLYGON)
        for v in layer:
          glVertex3f(v[0], v[2], v[1] * settings['separationFactor'])
        glEnd()

    if(settings['Faces']):
      glColor3f(0.8,0.2,0.3)
      glBegin(GL_TRIANGLES)
      for tri in modelView:
        glVertex3f(tri[0][0], tri[0][2], tri[0][1] * settings['separationFactor'])
        glVertex3f(tri[1][0], tri[1][2], tri[1][1] * settings['separationFactor'])
        glVertex3f(tri[2][0], tri[2][2], tri[2][1] * settings['separationFactor'])
      glEnd()

    if(settings['Lines']):
      glLineWidth(5)
      glColor3f(0.2,0,0)
      glBegin(GL_LINES)
      for tri in modelView:
        glVertex3f(tri[0][0], tri[0][2], tri[0][1] * settings['separationFactor'])
        glVertex3f(tri[1][0], tri[1][2], tri[1][1] * settings['separationFactor'])
        glVertex3f(tri[1][0], tri[1][2], tri[1][1] * settings['separationFactor'])
        glVertex3f(tri[2][0], tri[2][2], tri[2][1] * settings['separationFactor'])
        glVertex3f(tri[2][0], tri[2][2], tri[2][1] * settings['separationFactor'])
        glVertex3f(tri[0][0], tri[0][2], tri[0][1] * settings['separationFactor'])
      glEnd()

    if(settings['Points']):
      glPointSize(10)
      glColor3f(0.8, 0.7, 0.9)
      glBegin(GL_POINTS)
      for tri in modelView:
        glVertex3f(tri[0][0], tri[0][2], tri[0][1] * settings['separationFactor'])
        glVertex3f(tri[1][0], tri[1][2], tri[1][1] * settings['separationFactor'])
        glVertex3f(tri[2][0], tri[2][2], tri[2][1] * settings['separationFactor'])
      glEnd()

    pygame.display.flip()
    pygame.time.wait(16)

def updateView(ySeparation = settings['ySeparation'], verbose = False):
  global modelView
  global sliceView
  global sessionData
  contourList = sessionData['contourList']
  normalizedContourList = normalizeContourList(contourList, verbose)
  if(len(contourList) > 1):
    modelView = []
    sliceView = []
    genModels(normalizedContourList, ySeparation = ySeparation ,verbose = verbose)

def initSegmentation(verbose = False):
  global sessionData
  global settings
  sessionData['imgs'] = seg.readImages(dir = sessionData['imgsDir'], verbose = verbose)
  sessionData['contourList'] = []
  sessionData['rawContourList'] = []
  sessionData['initContour'] = seg.initializeContour(sessionData['imgs'][0], pointDistance= sessionData['contourInitPointDist'], verbose = verbose)
  settings['imgItr'] = 0
  settings['taskComplete'] = False
  initNewImage(verbose = verbose)

def initNewImage(verbose = False):
  global sessionData
  global settings
  if(settings['imgItr'] >= len(sessionData['imgs'])):
    print('Segmentation Complete')
    settings['taskComplete'] = True
    if(verbose):
      print('Saving')

    sessionData['modelView'] = modelView
    sessionData['sliceView'] = sliceView
    
    data = {}
    sessionDataToSave = ['sessionName', 'imgsDir' ,'modelView', 'sliceView']
    settingsToSave = ['Alpha', 'Beta', 'Gamma', 'maxItr', 'ySeparation']

    for x in sessionDataToSave:
      data[x] = sessionData[x]

    for x in settingsToSave:
      data[x] = settings[x]

    if(seg.writePolyData(data)):
      print('Successfully Written to file')
     
    else:
      print('Error Writing to file')

  else:
    sessionData['latestContour'] = sessionData['initContour']
    imgPath = sessionData['imgs'][settings['imgItr']]
    print('Processing Image:',imgPath)
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    sessionData['gMag'] = seg.calc_image_gradient(img)
  
def view(verbose = False, initSettings = {}, sessionConfig = {}):
  global settings
  global sessionData

  sessionData = {}
  if 'sessionName' in sessionConfig:
    for k in sessionConfig:
      sessionData[k] = sessionConfig[k]
  else:
    sessionData['sessionName'] = 'test'
    sessionData['polyDataFile'] = 'data/polyData.json'
    sessionData['imgsDir'] = 'data/brain'
    sessionData['contourInitPointDist'] = 25

  if('Alpha' in initSettings):
    for k in initSettings:
      settings[k] = initSettings[k]
  else:
    settings['Alpha'] = 0.1
    settings['Beta'] = 3
    settings['Gamma'] = 0.05
    settings['maxItr'] = 200
    settings['ySeparation'] = 0.05
  runLoop(verbose = verbose)

def loadPoly(sessionName, verbose = False):
  if(verbose):
    print('Loading PolyData')

def loadPreprocessed(sessionName, verbose = False):
  global modelView
  global sliceView
  global sessionData
  global settings

  polyData = seg.readPolyData(verbose = verbose)
  if(verbose):
    print('Successfully Loaded PolyData with',len(polyData['data']), 'Models')

  i = seg.findSessionIndex(sessionName, polyData['data'])
  if(i == -1):
    print('Session Not found')

  else:
    x = polyData['data'][i]
    modelView = x['modelView']
    sliceView = x['sliceView']
    if(verbose):
      print('Loading',sessionName)
    
    settings['taskComplete'] = True
    settings['imgItr'] = 0
    sessionData['imgsDir'] = x['imgsDir']

    print('-------------------------')
    print('Configuration')
    for k in x:
      s = x[k]
      if(type(s) == list):
        s = 'List of Length ' + str(len(s))
      print(k,':', s)
    print('-------------------------')

    runLoop(verbose = verbose)

runSettings = {}
runSettings['Alpha'] = 1
runSettings['Beta'] = 1
runSettings['Gamma'] = 1
runSettings['maxItr'] = 200
runSettings['ySeparation'] = 0.05

runSessionConfig = {}
runSessionConfig['sessionName'] = 'brain'
runSessionConfig['polyDataFile'] = 'data/polyData.json'
runSessionConfig['imgsDir'] = 'data/brain'
runSessionConfig['contourInitPointDist'] = 20

#view(verbose = True, initSettings = runSettings, sessionConfig = runSessionConfig)
loadPreprocessed('brain', verbose = True)