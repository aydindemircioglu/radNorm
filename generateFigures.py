import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw, ImageFont
import shutil



def addText (finalImage, text = '', org = (0,0), fontFace = "Arial", fontSize = 12, color = (255,255,255)):
     # Convert the image to RGB (OpenCV uses BGR)
     #tmpImg = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
     tmpImg = finalImage
     pil_im = Image.fromarray(tmpImg)
     draw = ImageDraw.Draw(pil_im)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font, fill = color)
     #tmpImg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
     tmpImg = np.array(pil_im)
     return (tmpImg.copy())



def addBorder (img, pos, thickness):
    if pos == "H":
        img = np.hstack([255*np.ones(( img.shape[0],int(img.shape[1]*thickness), 3), dtype = np.uint8),img])
    if pos == "V":
        img = np.vstack([255*np.ones(( int(img.shape[0]*thickness), img.shape[1], 3), dtype = np.uint8),img])
    return img



def generateFigure2(m = 128, fs = 96):
    f3a = cv2.imread("./results/Fig3a.png")
    f3b = cv2.imread("./results/Fig3b.png")
    f3c = cv2.imread("./results/Fig3c.png")
    f3a = cv2.cvtColor(f3a, cv2.COLOR_RGB2BGR)
    f3b = cv2.cvtColor(f3b, cv2.COLOR_RGB2BGR)
    f3c = cv2.cvtColor(f3c, cv2.COLOR_RGB2BGR)
    f3a = addText (f3a, "a", (96,200), "Arial", fs, color=(0,0,0))
    f3b = addText (f3b, "b", (96,200), "Arial", fs, color=(0,0,0))
    f3c = addText (f3c, "c", (96,200), "Arial", fs, color=(0,0,0))

    w = f3a.shape[1] + m + f3b.shape[1]
    w = np.max ([w, f3c.shape[1]])

    h = np.max([f3a.shape[0], f3b.shape[0]])
    h = h + m + f3c.shape[0]

    z = np.zeros((h,w,3), dtype = np.uint8)+255

    # place 3
    d = (w-f3c.shape[1])//2
    z[h-f3c.shape[0]:h,d:d+f3c.shape[1],:] = f3c

    Image.fromarray(z)



    # place 1
    d = (w-f3b.shape[1]-f3a.shape[1])//2
    my = d-f3c.shape[0]-3*m//2
    z[my-f3a.shape[0]:my, d:d+f3a.shape[1],:] = f3a

    # place 2
    #d = (w-d-d-f3b.shape[1])//2
    d = (w-f3b.shape[1]-f3a.shape[1])//2
    my = d-f3c.shape[0]-3*m//2
    z[my-f3b.shape[0]:my, w-d-f3b.shape[1]:w-d,:] = f3b

    z = cropImage(z)
    z = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./paper/Figure_2.png", z)


def cropImage (img):
    for y0 in range(img.shape[0]):
        if np.sum(img[y0,:,:] - img[y0,0,:]) > 0:
            break
    for x0 in range(img.shape[1]):
        if np.sum(img[:,x0,:] - img[0,x0,:]) > 0:
            break
    for x1 in range(img.shape[1]-1,0,-1):
        if np.sum(img[:,x1,:] - img[0,x1,:]) > 0:
            break
    img = img[y0:, x0:x1, :]
    return img



def generateFigure2():
    shutil.copyfile ("./results/Fig3a.png", "./paper/Figure_2.png")


def generateFigure3(m = 128, fs = 96):
    f3a = cv2.imread("./results/Fig3b.png")
    f3b = cv2.imread("./results/Fig3c.png")
    f3a = cv2.cvtColor(f3a, cv2.COLOR_RGB2BGR)
    f3b = cv2.cvtColor(f3b, cv2.COLOR_RGB2BGR)
    f3a = addText (f3a, "a", (96,200), "Arial", fs, color=(0,0,0))
    f3b = addText (f3b, "b", (96,200), "Arial", fs, color=(0,0,0))

    h = f3a.shape[0] + m + f3b.shape[0]
    w = np.max([f3a.shape[1], f3b.shape[1]])
    z = np.zeros((h,w,3), dtype = np.uint8)+255

    # place 1
    d = (w - f3a.shape[1])//2
    z[0:f3a.shape[0], d:d+f3a.shape[1],:] = f3a

    # place 2
    d = (w - f3b.shape[1])//2
    z[m+f3a.shape[0]:h, d:d+f3b.shape[1],:] = f3b
    Image.fromarray(z)

    z = cropImage(z)
    z = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./paper/Figure_3.png", z)



def generateFigure4(m = 128, fs = 96):
    f4a = cv2.imread("./results/Fig4a.png")
    f4b = cv2.imread("./results/Fig4b.png")
    f4a = cv2.cvtColor(f4a, cv2.COLOR_RGB2BGR)
    f4b = cv2.cvtColor(f4b, cv2.COLOR_RGB2BGR)
    f4a = addText (f4a, "a", (96,200), "Arial", fs, color=(0,0,0))
    f4b = addText (f4b, "b", (96,200), "Arial", fs, color=(0,0,0))

    h = np.max ([f4a.shape[0], f4b.shape[0]])
    w = f4a.shape[1] + m + f4b.shape[1]
    z = np.zeros((h,w,3), dtype = np.uint8)+255
    z[h-f4a.shape[0]:,0:f4a.shape[1],:] = f4a

    z[h-f4b.shape[0]:, f4a.shape[1]+m:f4a.shape[1]+m+f4b.shape[1],:] = f4b

    z = cropImage(z)
    z = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./paper/Figure_4.png", z)



def generateFigure5(m = 128, fs = 96):
    f3a = cv2.imread("./results/Fig5.png")
    f3b = cv2.imread("./results/Fig6.png")
    f3a = cv2.cvtColor(f3a, cv2.COLOR_RGB2BGR)
    f3b = cv2.cvtColor(f3b, cv2.COLOR_RGB2BGR)
    f3a = addText (f3a, "a", (96,200), "Arial", fs, color=(0,0,0))
    f3b = addText (f3b, "b", (96,200), "Arial", fs, color=(0,0,0))

    h = f3a.shape[0] + m + f3b.shape[0]
    w = np.max([f3a.shape[1], f3b.shape[1]])
    z = np.zeros((h,w,3), dtype = np.uint8)+255

    # place 1
    d = (w - f3a.shape[1])//2
    z[0:f3a.shape[0], d:d+f3a.shape[1],:] = f3a

    # place 2
    d = (w - f3b.shape[1])//2
    z[m+f3a.shape[0]:h, d:d+f3b.shape[1],:] = f3b
    Image.fromarray(z)

    z = cropImage(z)
    z = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./paper/Figure_5.png", z)



def generateFigureS1():
    shutil.copyfile ("./results/FigS1_0.png", "./paper/Supplementary_Figure_1a.png")
    shutil.copyfile ("./results/FigS1_1.png", "./paper/Supplementary_Figure_1b.png")
    shutil.copyfile ("./results/FigS1_2.png", "./paper/Supplementary_Figure_1c.png")
    shutil.copyfile ("./results/FigS2.png", "./paper/Supplementary_Figure_2.png")

if __name__ == '__main__':
    generateFigure2()
    generateFigure3()
    generateFigure4()
    generateFigure5()
    generateFigureS1()


#
