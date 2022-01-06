
import pdb
import cv2
import pytesseract
import numpy as np
import os

path = "C:/Users/Daniel/OneDrive/Documents/coding/irish_snap_ai/images"

cap = cv2.VideoCapture(0)

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image, 5)

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result 

def find_cards(image):
    contours, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    #print(len(contours))
    
    contours_sort = []
    hierarchy_sort = []
    cnt_is_card = np.zeros(len(contours),dtype=int)

    for i in index_sort:
        contours_sort.append(contours[i])
        hierarchy_sort.append(hierarchy[0][i])

    for i in range(len(contours_sort)):
        size = cv2.contourArea(contours_sort[i])
        peri = cv2.arcLength(contours_sort[i],True)
        approx = cv2.approxPolyDP(contours_sort[i],0.01*peri,True)
        
        if ((size > 3200) and (size<3300)
            and (hierarchy_sort[i][3] == -1) and (len(approx) == 4)):

            cnt_is_card[i] = 1
            #x,y,w,h = cv2.boundingRect(contours_sort[i])
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            
    return contours_sort, cnt_is_card

def preprocess(contour, image):
    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)

    x,y,w,h = cv2.boundingRect(contour)

    

    new_img = flatterner(image, pts, w, h)
    return new_img
    

def flatterner(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        

    return warp

cards = []
k = 0
while True:
    

    
    card_input = input("Number")
    card_rank = card_input.split()[0]
    #card_suit = card_input.split()[-1]
    #print(card_rank)
    #print(card_suit)
    if card_rank != '':
        ret, frame = cap.read()
        frame_grey = get_grayscale(frame)
            

        frame_blur = cv2.GaussianBlur(frame_grey, (5,5),0)

        edged = cv2.Canny(frame_blur, 30, 200)
        thresh = cv2.threshold(edged, 210,230, cv2.THRESH_BINARY)[1]
            
            
        print("qww")    
        contours_sort, cnt_is_card = find_cards(thresh)
        #print(cnt_is_card)
        #print(contours_sort)

        if len(contours_sort) != 0:
                
            for i in range(len(contours_sort)):
                if (cnt_is_card[i] == 1):
                    print("hello!1")
                    card = preprocess(contours_sort[i], frame)
                    print("hello")
                    s = cv2.imwrite(os.path.join(path, f'{card_rank}_of_hearts_3.png'), card)
                        
        else:
            pass

    cv2.imshow("frame", frame)
    k +=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Escaping")
        break

#cap.release()
#cv2.destroyAllWindows()