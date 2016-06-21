# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:52:04 2016

@author: David
"""

import csv
import cv2
import cv2.cv as cv
import numpy as np
from numpy import linalg as LA
import fnmatch
import os
import matplotlib.pyplot as plt
import scipy.fftpack


def rescale(A):
    [c,r] = A.shape;
    mean = A[0,:];#Assign the first image as the MEAN image
    error = 10;
    
    old = np.zeros((1,80))
    j=0
    while error>0.00001:
        scale = np.sqrt(np.sum(np.square(mean)))
        
        A = np.divide(A,scale)
        mean = np.divide(mean,scale)
        old[:] = mean[:];
        
        for i in range(c):#For for calculating the rotation of the matrices.
            #plt.plot(A[i,::2],A[i,1::2])
            #plt.plot(mean[::2],mean[1::2])
            
            s, a, T, A[i,:]= transform(A[i,:], mean);
            #plt.plot(A[i,::2],A[i,1::2])
            A[i,:] = project_tangent(A[i,:], mean); #Rotating the matrix X coordinate
            #plt.plot(A[i,::2],A[i,1::2])
            #plt.show()
            
        mean[:] = np.mean(A,0)#Create a new mean matrix based on the mean of the rotated and scaled shapes.
        scale = np.sqrt(np.sum(np.square(mean)));
        mean[:] = mean /scale;
        error = np.linalg.norm(mean-old);
        #print j
        #print error
        
    mean = mean*scale;
        
    return mean,A,error


def project_tangent(A, T):
    '''
    Project onto tangent space
    @param A:               
    @param T:                            
    @return: s = scaling, alpha = angle, T = transformation matrix
    '''
    tangent = np.dot(A, T);
    A_new = A/tangent;
    return A_new


def transform(A, T):
    '''
    Calculate scaling and theta angle of image A to target T
    @param A:               
    @param T:                            
    @return: s = scaling, alpha = angle, Tr = transformation matrix
    '''
    Ax, Ay = split(A)
    Tx, Ty = split(T)
    #b2 = (np.dot(Ax, Ty)-np.dot(Ay, Tx))/np.power(np.dot(A, A),2)
    #a2 = np.dot(T, A)/np.power(np.dot(A, A),2)
    
    b2 = (np.dot(Ax, Ty)-np.dot(Ay, Tx))/np.dot(A.T, A)
    a2 = np.dot(T, A)/np.dot(A.T, A)
        
    alpha = np.arctan(b2/a2) #Optimal angle of rotation is found.
    Tr = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    s =np.sqrt(np.power(a2,2) + np.power(b2,2))
    
    result = np.dot(s*Tr, np.vstack((Ax,Ay)));
    #plt.plot(result[0,:],result[1,:])
    new_A = merge(result);
    Tr = s*Tr
    
    return s, alpha, Tr, new_A
    
def split (A):
    x = A[::2];#Divide in X coordinates
    y = A[1::2];#Divide in Y coordinates
    return x,y

def merge(XY):
    A = np.zeros((1,XY.shape[1]*2))
    A[0,::2] = XY[0, :] ;
    A[0,1::2] = XY[1, :];
    return A

def PCA(X,Variation):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param Variance:         Proportion of the total variation desired.                        
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape 
    Xm = np.mean(X, axis=0)
    x = np.zeros((n,d))
    x = X - Xm
    x = x.T
    Xc = np.dot(x.T,x)
    [L,V] = LA.eig(Xc)
    [ne] = L.shape
    index = np.argsort(-np.absolute(L))
    Li = L[index]
    varTot = np.sum(np.absolute(Li))
    varSum = 0;
    for numEig in range(0,ne):
        varSum = varSum + np.absolute(Li[numEig])
        #print varSum
        #print varTot
        if varSum/varTot >= Variation:
            print 'Number of Eigenvectors after PCA ='
            print numEig
            break
    Vi = np.dot(x,V)
    
    Vii = Vi[:,index]
    Viii = Vii[:,:numEig]
    Liii = Li[:numEig]
    
    VI = np.dot(Viii.T,Viii)
    #print(np.diagonal(VI))
    print(Viii)
    VV= np.divide(Viii,np.sqrt(np.diagonal(VI)))
    print("after")
    print(VV)
    
    print("pca norm")
    print(np.sum(VV, axis=0))
    
    #for ii in range(0,Viii.shape[1]):
    #    sumi = 0;
    #    for kk in range(0,Viii.shape[0]):
    #        sumi = Viii[kk,ii] + sumi;
    #    print(sumi)
    #    for jj in range(0,Viii.shape[0]):
    #        Viii[jj,ii] = Viii[jj,ii]/sumi
    #print Viii.shape
    
    #print(Viii[0])
    print("Eigs on top")
    return [Liii,VV,Xm]
    
def Matching_Real(initialPossition,eigVals,eigVecs,mean,testImage):
    '''We do the PCA and the model "creation" using the form [X1,Y1,X2,Y2...], not sepparated
    initialPossition = Initial Possition for the model as [Xt,Yt]
    
    '''
    
    b = np.zeros((eigVecs.shape[1],1))
    
    Tr = np.identity(2)
    print("Matching-----------")
    error = 1000;
    repetitions = 0;
    while repetitions<3:#error > 0.0001:
        repetitions = repetitions +1;
        print(repetitions)
        
        X = np.add(mean, np.dot(eigVecs,b).T)
        
        Xx,Xy = split(X.T)
   
        Xin = np.vstack((np.add(initialPossition[0],np.dot(Tr,np.hstack((Xx,Xy)).T)[0,:]),np.add(initialPossition[1],np.dot(Tr,np.hstack((Xx,Xy)).T)[1,:])))
        
        tes = testImage.reshape(height,width)
        test = tes.copy()
        #for i in range(40):
        #    test[Xin[1,i],Xin[0,i]]=255;
        
        
        XinM = merge(Xin)
        print(XinM)
        Xrec = mahalanobisMatching(XinM.T,testImage)

        
        
        #for i in range(40):
            #test[Xrec[i*2+1],Xrec[i*2]]=255;
        
        #cv2.imshow('test', test.astype(np.uint8));
        #cv2.waitKey(0);
        
        initialPossition = np.mean(split(Xrec)-Xin, axis=1)

        s, a, Tr, xFin = transform(XinM[0], Xrec)
        

        Tinv = np.linalg.inv(Tr)

        y = np.dot(Tinv,np.vstack((np.subtract(np.vstack((split(Xrec)))[0],initialPossition[0]),np.subtract(np.vstack(split(Xrec))[1],initialPossition[1]))))
        
        print(merge(y))
        
        print(eigVals)       
        
        bn = np.dot(eigVecs.T,np.subtract(merge(y),mean).T)
        for i in range(bn.shape[0]):
            if bn[i] >= 3*np.sqrt(eigVals[i]):
                b[i] = 3*np.sqrt(eigVals[i])
            elif bn[i] <= -3*np.sqrt(eigVals[i]):
                b[i] = -3*np.sqrt(eigVals[i])
            else:
                b[i] = bn[i]
        error = np.linalg.norm(X - mean + np.dot(eigVecs,b))#not calculated on the right place, but might work, should be on the image space
        
    
    cv2.destroyAllWindows()
    return
    
def model_learning(t_size,data):
    
    # split dataset in Target and Training set
    # Do it for all 
    [c,r] = data.shape
    target = np.zeros((t_size, 80))
    training = np.zeros((c-t_size, 80))
    for i in range(c/t_size):
        start = i*t_size
        stop =  i+t_size
        target[:,:]= data[start:stop, :]
        training[:,:] = data
        model, A, error = rescale(training);
        [eigVals, eigVecs, mean] = PCA(model,0.98);
        
        for i in range(t_size):
            result = Matching(target,eigVals,eigVecs,mean);
            print result

    return model, error
    
 
def butt(image, f, n=2, pxd=0.5):
    """Designs an n-th order lowpass 2D Butterworth filter with cutoff
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
   
    pxd = float(pxd)
    rows, cols = image.shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = 1 / (1.0 + (f / radius)**(2*n))
    return filt
    
def gaus(img, sigma):
    
    # Number of rows and columns
    [rows, cols] = img.shape
    
    # Create Gaussian mask of sigma = 10
#    M = 2*rows + 1
#    N = 2*cols + 1
    M = rows
    N = cols
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
    filt = np.exp(-gaussianNumerator / (2*sigma*sigma))
    
    return filt 
    
def nothing(x):
    pass
    
def preproc(img):
    
    img = cv2.imread("Project_Data/_Data/Radiographs/01.tif",0)
    [rows, cols] = img.shape
    
    heq = cv2.equalizeHist(img)
        
    cv2.namedWindow('can')
    # create trackbars for color change
    cv2.createTrackbar('Max','can',0,255, nothing)
    cv2.createTrackbar('Min','can',0,255, nothing)
    cv2.createTrackbar('sigma','can',0,255, nothing)
    cv2.createTrackbar('cut','can',0,255, nothing)    
    #cv2.createTrackbar('kern_size','can',1,31, nothing)
    edges = img;
    #lapl = edges;
    while(1):
        cv2.imshow('can',edges)
        #k = cv2.waitKey(1) & 0xFF
        #if k == 27:
        #    break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # get current positions of four trackbars
        r = cv2.getTrackbarPos('Max','can')
        g = cv2.getTrackbarPos('Min','can')
        s = cv2.getTrackbarPos('sigma','can')
        
        c = cv2.getTrackbarPos('cut','can')
        #kers = cv2.getTrackbarPos('kern_size','can')
        # Convert image to 0 to 1, then do log(1 + I)
        imgLog = np.log1p(np.array(img, dtype="float") / 255)
        
        # Low pass and high pass filters
        Hlow = gaus(img,s/2)
        Hhigh = butt(img,c)
        
        # Move origin of filters so that it's at the top left corner to
        # match with the input image
        
        HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
        HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
        
        # Filter the image and crop
        If = scipy.fftpack.fft2(imgLog.copy(), (rows,cols))
        Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (rows,cols)))
        #new = cv2.equalizeHist(img)
        If2 = scipy.fftpack.fft2(Iouthigh.copy(), (rows,cols))
        Ioutlow = scipy.real(scipy.fftpack.ifft2(If2.copy() * HlowShift, (rows,cols)))
        

        
        # Anti-log then rescale to [0,1]
        Ihmf = np.expm1(Iouthigh)
        Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
        Ihmf2 = np.array(255*Ihmf, dtype="uint8")
        
        Ihf = np.expm1(Ioutlow)
        Ihf = (Ihf - np.min(Ihf)) / (np.max(Ihf) - np.min(Ihf))
        Ihf2 = np.array(255*Ihf, dtype="uint8")
        edges = cv2.Canny(Ihf2,g,r)
        
        #lapl = cv2.Laplacian(edges,cv2.CV_64F,ksize=kers)
        
        
    cv2.destroyAllWindows()

   

    # Show all images
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    cv2.imshow('Homomorphic Filtered Result 1', Ihmf2)
    cv2.waitKey(0)
    cv2.imshow('Homomorphic Filtered Result 2', heq)
    cv2.waitKey(0)
    cv2.imshow('Homomorphic Filtered Result', Ihf2)
    cv2.waitKey(0)
    cv2.imshow('Homomorphic Filtered Result', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print r,g,s,c
    
    return r,g,s,c

def preproc_real(img,r,g,s,c):
    [rows, cols] = img.shape

    #heq = cv2.equalizeHist(img)
        
    #cv2.namedWindow('can')

    edges = img;
    
    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)
    
    # Low pass and high pass filters
    Hlow = gaus(img,s/2)
    Hhigh = butt(img,c)
        
    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
    
    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (rows,cols))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (rows,cols)))
    #new = cv2.equalizeHist(img)
    If2 = scipy.fftpack.fft2(Iouthigh.copy(), (rows,cols))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If2.copy() * HlowShift, (rows,cols)))
    

    
    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iouthigh)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")
    
    Ihf = np.expm1(Ioutlow)
    Ihf = (Ihf - np.min(Ihf)) / (np.max(Ihf) - np.min(Ihf))
    Ihf2 = np.array(255*Ihf, dtype="uint8")
    edges = cv2.Canny(Ihf2,g,r)   


    
    return edges
    
def test_imagefilter():
    #img = cv2.imread("C:\\Users\\David\\Google Drive\\KULeuven\\Computer vision\\Nieuwe map\\_Data\\Radiographs\\01.tif",0);
    img = cv2.imread("Project_Data/_Data/Radiographs/01.tif",0);
    
    rows, cols = img.shape
    print rows,cols
    img2 = img[400:1300, 1000:1900]
    img3 = img[100:1500,60:2930]
    #r,g,s,c = preproc(img2)
    #preproc_real(img2,r,g,s,c)
    cv2.imshow('Transformed Image', preproc_real(img2,29, 42, 100, 12))
    cv2.waitKey(0)
    cv2.imshow('Transformed Image', preproc_real(img3,29, 42, 100, 12))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def image_fit(img, s_model, i_model,  l):
    
    ns = 0;
    n = 5;
    while ns<0.9:   
        for i in range(s_model):
            sobelx = 2;
            sobely = 2;
            x_model = s_model[i]+sobelx *  np.linspace(-n, 1, n)
            y_model = s_model[i]+sobely * np.linspace(-n, 1, n)
            g = img[x_model, y_model]
            dist = mahalanobis(X, g)
    return T, s,a 

def extract_Features(A,n,imgR):
    """A should only be one vector, e.g. reader[0]
        n is the profile size, number of pixels in the lines
        imgR is the corresponding image to extract features in vector form.
        NOTE, for some reason we have 80 points, but we only obtain 40 vectors displayed in the image."""
   
#    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) 
#    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    img = imgR.reshape(height,width);

    #img = cv2.imread("Project_Data/_Data/Radiographs/01.tif",0)
    #cv2.imshow('test', img);
    #img = cv2.imread("C:\\Users\\David\\Google Drive\\KULeuven\\Computer vision\\Nieuwe map\\_Data\\Radiographs\\01.tif",0);
    np.linspace(0,80,41)
    #n=50;
    vec = np.zeros((40*2,(2*n+1)));
    vecExtr = np.zeros(((vec.shape[1]),40));
    for j in range(40):
        x1= A[np.mod(j*2-2,80)];
        y1 = A[np.mod(j*2-1, 80)];
        x2 = A[np.mod(j*2,80)];
        y2 = A[np.mod(j*2+1,80)];
        x3 = A[np.mod(j*2+2,80)];
        y3 = A[np.mod(j*2+3,80)];
        
        
        
        dx = x3-x1;
        dy = y3-y1;
        
        mag = np.sqrt(dx*dx+dy*dy);
         
        dx = dx/mag;
        dy = dy/mag;
        
        x = x2;
        y = y2;
        
        #print x,y
        
        nx = -dy;
        ny = dx;
        length = np.linspace(-n,n,2*n+1);

        vec[2*j,:] = x + length * nx;
        vec[2*j+1,:] = y + length * ny;
        for i in length:
            vecExtr[i,j] = img[vec[2*j+1,i],vec[2*j,i]];

#    cv2.imshow('test', img.astype(np.uint8));
#    cv2.waitKey(0);
#    cv2.destroyAllWindows()
    return vec,vecExtr
            
def distribution_Training(train,profileSize):
    
    vec,_ = extract_Features(reader[0],profileSize,train[0]);
    A = np.zeros((train.shape[0],vec.shape[1]*vec.shape[0]/2))
    for i in range(train.shape[0]):
        #vecExtr = np.zeros(((vec.shape[1]),80));
        img = train[i,:].reshape(height,width);
        lapl = cv2.Laplacian(img,cv2.CV_64F,ksize=3);
        vec,vecExtrP = extract_Features(reader[i*8],profileSize,lapl);
#        cv2.imshow('test', lapl.astype(np.uint8));
#        cv2.waitKey(0);
#        cv2.destroyAllWindows()
#        print(np.sum(vec))
#        for j in range(80):
#            for k in range(vec.shape[1]):
#                vecExtr[k,j] = lapl[vec[2*j,k],vec[2*j+1,k]];
        for l in range(vecExtrP.shape[1]):
            A[i,vecExtrP.shape[0]*l:vecExtrP.shape[0]*(l+1)] = vecExtrP[:,l];
            
#    print(vecExtrP.shape[0])
#    print(A.shape)
    return A   


    
def mahalanobisMatching(testModel,testImage):
    '''
    Calculates the best point locations based on mahalanobis distance.
    @param testModel: Test shape
    @param testImage: Image to test the shape on
    '''
    
    #testImage = training[0]
    
    profileSize = 10
    comparedProfileSize = 5
    
    sampledProfile = distribution_Training(training,comparedProfileSize)
    #print(sampledProfile.shape)
    lapl = cv2.Laplacian(testImage,cv2.CV_64F,ksize=3);
    vec,vecExtrP = extract_Features(testModel,profileSize,lapl)
    subProfile = np.zeros((sampledProfile.shape[0],(2*comparedProfileSize+1)))
    #print(subProfile.shape)
    newPointCoords = np.zeros((40*2))
    for k in range(40):
        subProfile = sampledProfile[:,k:k+(2*comparedProfileSize+1)]
        #print(subProfile.shape)
        index = 0;
        prevMaha = 1000000;
        for i in range(2*profileSize+1 - 2*comparedProfileSize+1-1):
            #print(i,i+(2*comparedProfileSize+1))
            #print(vecExtrP[:,k].shape)
            #print(vecExtrP[i:i+(2*comparedProfileSize+1),k].shape)
            m = mahalanobis(subProfile,vecExtrP[i:i+(2*comparedProfileSize+1),k])
            if (m<prevMaha):
                prevMaha = m;
                index = i;
        newPointCoords[2*k] = vec[2*k,comparedProfileSize+index]
        newPointCoords[2*k+1] = vec[2*k+1,comparedProfileSize+index]

    return newPointCoords
    
def mahalanobis(X, g):
    """"Calculate the Mahalanobis distance based on 
    @param X   matrix of training data and shape = (nb samples, nb dimensions of each sample)
    @param g   the current samplepoints around the model  
    
    [rows, cols] = X.shape with,
    Rows: Instance
    Cols: pixel intensity
    
    @param return Mahalanamobis distance
    """
    
    [rows, cols] = X.shape 
    Xm = np.mean(X, axis=0)
    
    #[n,d] = X.shape 
    #Xm = np.mean(X, axis=0)
    #x = np.zeros((n,d))
    #x = X - Xm
    #x = x.T
    #Xc = np.dot(x.T,x)
    
    x = np.zeros((rows, cols))
    x = X-Xm;
    #xcov = np.cov(Xp)
    
    x = x.T
    xcov = np.dot(x,x.T)
    
    i_xcov = np.linalg.pinv(xcov)
    g = g-Xm
    dist = (g).dot(i_xcov).dot(g.T);
    
    return dist    
    
def test_profile_grad( A):
    
#    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) 
#    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    for j in range(80):
        x1 = A[0,np.mod(j*2, 80)]
        y1= A[0,np.mod(j*2+1,80)];
        x2 = A[0,np.mod(j*2+2,80)];
        y2 = A[0,np.mod(j*2+3,80)];
        
        #print x1, y1, x2,y2;
        
#        x_l = np.linspace(-10, 1, 10)*sobelx[x,y]/np.sqrt(np.power(sobelx[x,y],2)+np.power(sobely[x,y],2))
#        y_l = np.linspace(-10, 1, 10)*sobely[x,y]/np.sqrt(np.power(sobelx[x,y],2)+np.power(sobely[x,y],2))
#        print x
#        print y
#        print np.rint(gx)
#        print np.rint(gy)
    
    cv2.imshow('test', img);
    cv2.waitKey(0);
    cv2.destroyAllWindows()
    
def multi_resol(img, A):
    
    L = 3
    while (L>0):
        [T,s,a] = image_fit(img, A, L)
        L=L-1;
    return T,s,a

if __name__ == '__main__':
    
    global height 
    height = 1400;
    global width 
    width = 2870;
    
    reader = np.zeros([112,80])
    readerP = np.zeros([112,80])
    i=0;
    #directory = "C:\Users\David\Google Drive\KULeuven\Computer vision\Nieuwe map\\_Data/Landmarks/original/"
    directory = "Project_Data/_Data/"
    for filename in fnmatch.filter(os.listdir(directory + "Landmarks/original/"),'*.txt'):
        reader[i,:] = np.loadtxt(open(directory+ "Landmarks/original/"+filename,"rb"),delimiter=",",skiprows=0)
        reader[i,::2]  = reader[i,::2]-60 #-np.mean(reader[i,::2]);#Zero-mean of the X axis
        reader[i,1::2] = reader[i,1::2]-100#-np.mean(reader[i,1::2]);#Zero-mean of the Y axis
        
        readerP[i,:] = reader[i,:];
        readerP[i,::2]  = reader[i,::2]-np.mean(reader[i,::2]);#Zero-mean of the X axis
        readerP[i,1::2] = reader[i,1::2]-np.mean(reader[i,1::2]);#Zero-mean of the Y axis
        
        i+=1;
        
    vSize = height*width;
    training = np.zeros((14,vSize))#, dtype=np.int)
    i = 0;
    for filename in fnmatch.filter(os.listdir(directory + "Radiographs/"),'*.tif'):
        img = cv2.imread(directory + "Radiographs/" + filename,0)
        
        img2 = img.copy()
        cv2.rectangle(img2, (60,100), (2930, 1500), (0, 255, 0), 2)
        result = cv.GetSubRect(cv.fromarray(img2), (60, 100, int(2930-60),int(1500-100)))
        result = np.asarray(result)

        imgT = np.zeros((1,vSize), dtype=np.int)
        imgT = result.reshape(1,vSize)
        training[i] = imgT
#        break;
        i+=1;
        
    
        
    #print reader[::8,:]
    shape, A,error = rescale(readerP[::8,:]);
    [eigVals,eigVecs,mean] = PCA(A,0.98)
    
    
    initialPossition = [1362,888]
    
    Matching_Real(initialPossition,eigVals,eigVecs,mean,training[0])
    
    
    
#    model_learning(8,reader)

    #te = mahalanobisMatching(reader[0,:]+30)
    #for i in range(2):
    #    te = mahalanobisMatching(te)
    #    print(2-i);
    
    #te = np.round(te)
    #test = training[0].reshape(height,width)
    #for i in range(40):
    #    test[reader[0,i*2+1],reader[0,i*2]]=255;
     #   test[te[i*2+1],te[i*2]]=0;
    #cv2.imshow('test', test.astype(np.uint8));
    #cv2.waitKey(0);
    #cv2.destroyAllWindows()


    
#    plt.plot(reader[0,::2],reader[0,1::2])
#   plt.plot(shape[::2],shape[1::2])
#   plt.show()
