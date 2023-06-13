from ultralytics import YOLO
import cv2
import numpy as np
model2 = YOLO('detection_best.pt')
model = YOLO('segmentation_best.pt')
results = model('6.jpg', save = False)
height, width, _ = cv2.imread('6.jpg').shape
# results5 = model2('6.jpg', save = False)
xplusy, xminusy = dict(), dict()
masks = results[0].masks.segments # Masks object for segmenation masks outputs for seg in masks [0]:
for seg in masks[0]:
    val1, val2 = seg [0] + seg [1], seg [0] - seg [1]
    xplusy[tuple([seg [0], seg[1]])] = val1
    xminusy [tuple([seg [0], seg [1]])] = val2

temp1, temp2, temp3, temp4 = min(xplusy.values()), max(xplusy.values()), min (xminusy.values()), max(xminusy.values()) 
res1 = [key for key in xplusy if xplusy [key] == temp1] 
res2 = [key for key in xplusy if xplusy [key] ==temp2]
res3 = [key for key in xminusy if xminusy [key] ==temp3]
res4 = [key for key in xminusy if xminusy [key] == temp4]
img = cv2.imread('6.jpg')
minxy= (int(res1[0][0]*width), int(res1[0] [1] *height)) 
maxxy = (int(res2 [0][0] *width), int(res2 [0] [1] *height)) 
minxmy = (int(res3 [0] [0] * width), int(res3 [0] [1] *height)) 
maxxmy = (int(res4 [0][0] *width), int (res4 [0] [1] *height))
color1 = (255, 0, 0)
color= (255, 0, 0)
color2 = (0, 0, 0)
color3 = (0,255,0)
color4 = (0,0,255)
# img = cv2.circle(img, (int(res1[0][0]*width), int (res1[0] [1] *height)), 5, color1, 22) 
# img = cv2.circle(img, (int(res2 [0][0] *width), int (res2 [0] [1] *height)), 5, color2, 22) 
# img = cv2.circle(img, (int(res3 [0][0]*width), int (res3 [0] [1] *height)), 5, color3, 22) 
# img = cv2.circle(img, (int(res4 [0][0] *width), int(res4 [0] [1] *height)), 5, color4, 22) 
# img = cv2.line (img, minxy, maxxmy, color, 5) 
# img = cv2.line (img, maxxy, minxmy, color, 5) 
# img= cv2.line (img, maxxy, maxxmy, color, 5) 
# img= cv2.line (img, minxy, minxmy, color, 5) 
boardTopLeft = [int(res1[0][0]*width), int (res1[0] [1] *height)]
boardBottomRight = [int(res2 [0][0] *width), int (res2 [0] [1] *height)]
boardBottomLeft = [int(res3 [0][0]*width), int (res3 [0] [1] *height)]
boardTopRight = [int(res4 [0][0] *width), int(res4 [0] [1] *height)]
img_copy = np.copy(img)
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
k = 3500
pt_A = [x-k for x in boardTopLeft]
pt_B = [boardBottomLeft[0] - k, boardBottomLeft[1] + k]
pt_C = [x+k for x in boardBottomRight] 
pt_D = [boardTopRight[0] + k, boardTopRight[1] - k] 
# print(pt_A)
# print(pt_A*2)
width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC)) 
 
 
height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD)) 

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])
M = cv2.getPerspectiveTransform(input_pts,output_pts)
out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
#out = img
cv2.imwrite("output2.jpg", out)
results = model('output2.jpg', save = False)
height, width, _ = cv2.imread('output2.jpg').shape
results5 = model2('output2.jpg', save = False)
xplusy, xminusy = dict(), dict()
masks = results[0].masks.segments # Masks object for segmenation masks outputs for seg in masks [0]:
for seg in masks[0]:
    val1, val2 = seg [0] + seg [1], seg [0] - seg [1]
    xplusy[tuple([seg [0], seg[1]])] = val1
    xminusy [tuple([seg [0], seg [1]])] = val2

temp1, temp2, temp3, temp4 = min(xplusy.values()), max(xplusy.values()), min (xminusy.values()), max(xminusy.values()) 
res1 = [key for key in xplusy if xplusy [key] == temp1] 
res2 = [key for key in xplusy if xplusy [key] ==temp2]
res3 = [key for key in xminusy if xminusy [key] ==temp3]
res4 = [key for key in xminusy if xminusy [key] == temp4]
img = cv2.imread("output2.jpg")
minxy= (int(res1[0][0]*width), int(res1[0] [1] *height)) 
maxxy = (int(res2 [0][0] *width), int(res2 [0] [1] *height)) 
minxmy = (int(res3 [0] [0] * width), int(res3 [0] [1] *height)) 
maxxmy = (int(res4 [0][0] *width), int (res4 [0] [1] *height))
color1 = (255, 0, 0)
color= (255, 0, 0)
color2 = (0, 0, 0)
color3 = (0,255,0)
color4 = (0,0,255)
img = cv2.circle(img, (int(res1[0][0]*width), int (res1[0] [1] *height)), 5, color1, 22) 
img = cv2.circle(img, (int(res2 [0][0] *width), int (res2 [0] [1] *height)), 5, color2, 22) 
img = cv2.circle(img, (int(res3 [0][0]*width), int (res3 [0] [1] *height)), 5, color3, 22) 
img = cv2.circle(img, (int(res4 [0][0] *width), int(res4 [0] [1] *height)), 5, color4, 22) 
img = cv2.line (img, minxy, maxxmy, color, 5) 
img = cv2.line (img, maxxy, minxmy, color, 5) 
img= cv2.line (img, maxxy, maxxmy, color, 5) 
img= cv2.line (img, minxy, minxmy, color, 5) 
boardTopLeft = [int(res1[0][0]*width), int (res1[0] [1] *height)]
boardBottomRight = [int(res2 [0][0] *width), int (res2 [0] [1] *height)]
boardBottomLeft = [int(res3 [0][0]*width), int (res3 [0] [1] *height)]
boardTopRight = [int(res4 [0][0] *width), int(res4 [0] [1] *height)]
cv2.imwrite("output3.jpg", img)
def inPiece(locations, coordinates, halfSL):
    # bool = False
    # bLeft1 = coordinates[0][0]
    # bRight1 = coordinates[1][0]
    # tLeft1 = coordinates[2][0]
    # tRight1 = coordinates[3][0]
    # pbLeft1 = locations[0][0]
    # pbRight1 = locations[1][0]
    # ptLeft1 = locations[2][0]
    # ptRight1 = locations[3][0]
    # bLeft2 = coordinates[0][1]
    # bRight2 = coordinates[1][1]
    # tLeft2 = coordinates[2][1]
    # tRight2 = coordinates[3][1]
    # pbLeft2 = locations[0][1]
    # pbRight2 = locations[1][1]
    # ptLeft2 = locations[2][1]
    # ptRight2 = locations[3][1]
    # return bool
    bLeft1 = coordinates[0][0]
    bRight1 = coordinates[1][0]
    tLeft1 = coordinates[2][0]
    tRight1 = coordinates[3][0]
    pbLeft1 = locations[0][0]
    pbRight1 = locations[1][0]
    ptLeft1 = locations[2][0]
    ptRight1 = locations[3][0]
    bLeft2 = coordinates[0][1]
    bRight2 = coordinates[1][1]
    tLeft2 = coordinates[2][1]
    tRight2 = coordinates[3][1]
    pbLeft2 = locations[0][1]
    pbRight2 = locations[1][1]
    ptLeft2 = locations[2][1]
    ptRight2 = locations[3][1]
    bool = False
    if(pbLeft1 <= bLeft1 + halfSL and pbLeft1 >= bLeft1 - halfSL and ptLeft1 <= tLeft1 + halfSL and ptLeft1 >= tLeft1 - halfSL and pbRight1 <= bRight1 + halfSL and pbRight1 >= bRight1 - halfSL and ptRight1 <= tRight1 + halfSL and ptRight1 >= tRight1 - halfSL) and (pbLeft2 <= bLeft2 + halfSL and pbLeft2 >= bLeft2 - halfSL and ptLeft2 <= tLeft2 + halfSL and ptLeft2 >= tLeft2 - halfSL and pbRight2 <= bRight2 + halfSL and pbRight2 >= bRight2 - halfSL and ptRight2 <= tRight2 + halfSL and ptRight2 >= tRight2 - halfSL): bool = True
    return bool
# x = (results[0].masks.segments[0][:,0]*W).astype("int")
# y = (results[0].masks.segments[0][:,1]*W).astype("int")
# print(x)
# boxes = results[0].boxes
# box = boxes[0]  # returns one box
# blk = np.zeros((H,W))
# blk[y,x] = 255
# cv2_imshow(blk.astype("uint8"))
# width = int(box.xywh[0][2])
# height = int(box.xywh[0][3])
tH1 =  int((((boardBottomRight[0]-boardTopRight[0])**2)+((boardBottomRight[1]-boardTopRight[1])**2))**0.5)
tH2 =  int((((boardBottomLeft[0]-boardTopLeft[0])**2)+((boardBottomLeft[1]-boardTopLeft[1])**2))**0.5)
tW1 = int((((boardTopLeft[0]-boardTopRight[0])**2)+((boardTopLeft[1]-boardTopRight[1])**2))**0.5)
tW2 = int((((boardBottomLeft[0]-boardBottomRight[0])**2)+((boardBottomLeft[1]-boardBottomRight[1])**2))**0.5)
if tH1 > tH2: height2 = tH1
else: height2 = tH2
if tW1 > tW2: width2 = tW1
else: width2 = tW2
if height2 < width2: totalSide = width2
else: totalSide = height2
sqLength = totalSide/8
# boardTopLeft = [int(box.xyxy[0][0]), int(box.xyxy[0][1])]
# boardBottomRight = [int(box.xyxy[0][2]), int(box.xyxy[0][3])]
# boardTopRight = [int(box.xyxy[0][0] )+ int(box.xywh[0][2]), int(box.xyxy[0][1])]
# boardBottomLeft = [int(box.xyxy[0][0]), int(box.xyxy[0][1]) + int(box.xywh[0][3])]
# # boardTopLeft = [int(box.xyxy[0][2]) - totalSide, int(box.xyxy[0][3]) - totalSide]
# # boardBottomRight = [int(box.xyxy[0][2]), int(box.xyxy[0][3])]
# # boardTopRight = [int(box.xyxy[0][2]) - totalSide, int(box.xyxy[0][3])]
# # boardBottomLeft = [int(box.xyxy[0][2]), int(box.xyxy[0][3]) - totalSide]
boxes = results5[0].boxes
names = []
boxCoords = dict()
for r in results5:
    for c in r.boxes.cls: names.append(model2.names[int(c)])
count = 0
for name in names: boxCoords[name] = []
for box in boxes:
    # boardTopLeft2 = [int(box.xyxy[0][2]) - totalSide, int(box.xyxy[0][3]) - totalSide]
    # boardBottomRight2 = [int(box.xyxy[0][2]), int(box.xyxy[0][3])]
    # boardTopRight2 = [int(box.xyxy[0][2]) - totalSide, int(box.xyxy[0][3])]
    # boardBottomLeft2 = [int(box.xyxy[0][2]), int(box.xyxy[0][3]) - totalSide]
    boardTopLeft2 = [int(box.xyxy[0][0]), int(box.xyxy[0][1])]
    boardBottomRight2 = [int(box.xyxy[0][2]), int(box.xyxy[0][3])]
    boardTopRight2 = [int(box.xyxy[0][0]) + int(box.xywh[0][2]), int(box.xyxy[0][1])]
    boardBottomLeft2 = [int(box.xyxy[0][0]), int(box.xyxy[0][1]) + int(box.xywh[0][3])]
    boxCoords[names[count]].append([boardBottomLeft2, boardBottomRight2, boardTopLeft2, boardTopRight2])
    count+=1
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
pieceDict = dict()
pieceLocations = dict()
currLocX = boardBottomLeft[0]
count = 0
for l in letters:
    currLocY = boardBottomLeft[1]
    count1 = count
    for x in range(1, 9):
        pieceDict[l + str(x)] = count1
        pieceLocations[l + str(x)] = [[currLocX, currLocY], [currLocX+sqLength, currLocY],[currLocX, currLocY-sqLength], [currLocX+sqLength, currLocY-sqLength]]
        currLocY -= sqLength
        count1 += 8
    count += 1
    currLocX += sqLength
finalState = dict()
for b in boxCoords:
    for b2 in boxCoords[b]: 
        for p in pieceLocations:
            if inPiece(b2, pieceLocations[p], sqLength/2):
                finalState[p] = b
                break
finalList = ['-' for x in range(0, 64)]
pieceDict2 = {'rook white':'R', 'pawn white': 'P', 'bishop white':'B', 'knight white': 'N', 'queen white':'Q', 'king white':'K', 'king black':'k', 'rook black':'r', 'pawn black': 'p', 'bishop black':'b', 'knight black': 'n', 'queen black':'q'}
for f in finalState: finalList[pieceDict[f]] = pieceDict2[finalState[f]]
s = ""
for f in finalList: s+=f
print(s)
# print(len(finalState))
# print(len(boxes))