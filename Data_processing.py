# -*- coding: utf-8 -*-
"""
@author: Sangbin

Data downloaded from https://www.cell.com/cms/attachment/2069440429/2067703065/mmc2.zip

This program extracts 
1. Neuromuscular function (movement)
2. Somatic investment (cross-sectional size)
3. Reproductive investment (cumulative area of eggs laid)
4. Lifespan
"""

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(77)

sPath = "C:/Users/Sangbin/Desktop/mmc2/processed/"

sFileList = os.listdir(sPath)

fMovementList = []
fBodySizeList = []
fEggList = []
fLifeSpanList = []

fEarlyMovementList = []
fEarlyBodySizeList = []
fEarlyEggList = []

# extract movement, body size, area of eggs laid, and lifespan data
for sFile in sFileList:
    
    if sFile != 'metadata.tsv':
        
        sInFile = open(sPath + sFile, "r")
        sInFile.readline()
        
        for sReadLine in sInFile.readlines():
            
            sReadList = sReadLine.replace("\n", "").split("\t")
            
            if float(sReadList[0]) < 2.1:
                
                fEarlyMovementList += [float(sReadList[22])]
                fEarlyBodySizeList += [float(sReadList[25])]
                fEarlyEggList += [float(sReadList[29])]
            
            try:
            
                fMovementList += [float(sReadList[22])]
                fBodySizeList += [float(sReadList[25])]
                fEggList += [float(sReadList[29])]
                
                if sReadList[0] == "15.3":
                    
                    fLifeSpanList += [15.3]
            
            except ValueError:
                
                fLifeSpanList += [float(sReadList[0])]
                
                sInFile.close()
                
                break

        sInFile.close()

# plot cohorts across disribution of adult lifespans
plt.xlabel('Adult Lifespan ($days$)')
plt.ylabel('Number of Individuals')
plt.hist(np.array(fLifeSpanList), bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], facecolor = 'C2')
plt.title("Cohorts Across Disribution of Adult Lifespans")
plt.show()

# plot cohorts across disribution of early adulthood movement
plt.xlabel('Displacement Over 3 Hours ($mm$)')
plt.ylabel('Frequency')
plt.hist(np.array(fEarlyMovementList), bins = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], facecolor = 'C4', density = True)
plt.title("Cohorts Across Disribution of Early-Adulthood Movement")
plt.show()

# plot cohorts across disribution of early adulthood cumulative area of eggs laid
plt.xlabel('Cumulative Area of Eggs Laid ($mm^2$)')
plt.ylabel('Frequency')
plt.hist(np.array(fEarlyEggList), bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], facecolor = 'C9', density = True)
plt.title("Cohorts Across Disribution of Early-Adulthood Cumulative Area of Eggs Laid")
plt.show()

# plot cohorts across disribution of early adulthood body size
plt.xlabel('Cross-Sectional Size (adjusted for machine bias) ($mm^2$)')
plt.ylabel('Frequency')
plt.hist(np.array(fEarlyBodySizeList), bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11], facecolor = 'C3', density = True)
plt.title("Cohorts Across Disribution of Early-Adulthood Body Size")
plt.show()

sOutFile12 = open(sPath + "../../Data/short_lived_train_accuracy.txt", "w") # for calcluating training accuracy
sOutFile13 = open(sPath + "../../Data/short_lived_test_accuracy.txt", "w") # for calculating test accuracy

sOutFile22 = open(sPath + "../../Data/normal_lived_train_accuracy.txt", "w") # for calcluating training accuracy
sOutFile23 = open(sPath + "../../Data/normal_lived_test_accuracy.txt", "w") # for calculating test accuracy

sOutFile32 = open(sPath + "../../Data/long_lived_train_accuracy.txt", "w") # for calcluating training accuracy
sOutFile33 = open(sPath + "../../Data/long_lived_test_accuracy.txt", "w") # for calculating test accuracy

nFileNum = 0
nRandList = np.random.choice(734, size = 367)

# classify extracted data into 27 classes and save
for sFile in sFileList:
    
    if sFile != 'metadata.tsv':
        
        sInFile = open(sPath + sFile, "r")
        sInFile.readline()
        
        sClassListString = ""
        
        for sReadLine in sInFile.readlines():
            
            sReadList = sReadLine.replace("\n", "").split("\t")
                            
            if float(sReadList[0]) <= 2:
                
                fMovement =  float(sReadList[22])
                fBodySize = float(sReadList[25])
                fEgg = float(sReadList[29])
                
                if fMovement < 0.438:
                    
                    if fBodySize < 0.0666:
                        
                        if fEgg < 0.0370:
                            
                            sClassListString += "1\t" # classified into class 1
                            
                        elif fEgg < 0.0775:
                            
                            sClassListString += "2\t" # classified into class 2
                            
                        else:
                            
                            sClassListString += "3\t" # classified into class 3
                        
                    elif fBodySize < 0.0809:
                        
                        if fEgg < 0.0370:
                            
                            sClassListString += "4\t" # classified into class 4
                            
                        elif fEgg < 0.0775:
                            
                            sClassListString += "5\t" # classified into class 5
                            
                        else:
                            
                            sClassListString += "6\t" # classified into class 6
                            
                    else:
                    
                        if fEgg < 0.0370:
                            
                            sClassListString += "7\t" # classified into class 7
                            
                        elif fEgg < 0.0775:
                            
                            sClassListString += "8\t" # classified into class 8
                            
                        else:
                            
                            sClassListString += "9\t" # classified into class 9
                            
                elif fMovement < 0.545:
                    
                    if fBodySize < 0.0666:
                        
                        if fEgg < 0.0370:
                        
                            sClassListString += "10\t" # classified into class 10    
                        
                        elif fEgg < 0.0775:
                        
                            sClassListString += "11\t" # classified into class 11    
                                                    
                        else:
                        
                            sClassListString += "12\t" # classified into class 12    
                                                
                    elif fBodySize < 0.0809:
                        
                        if fEgg < 0.0370:
                        
                            sClassListString += "13\t" # classified into class 13    
                            
                        elif fEgg < 0.0775:
                        
                            sClassListString += "14\t" # classified into class 14    
                            
                        else:
                        
                            sClassListString += "15\t" # classified into class 15    
                            
                    else:
                    
                        if fEgg < 0.0370:
                        
                            sClassListString += "16\t" # classified into class 16    
                            
                        elif fEgg < 0.0775:
                        
                            sClassListString += "17\t" # classified into class 17    
                            
                        else:
                        
                            sClassListString += "18\t" # classified into class 18    
                            
                else:
                    
                    if fBodySize < 0.0666:
                        
                        if fEgg < 0.0370:
                        
                            sClassListString += "19\t" # classified into class 19    
                            
                        elif fEgg < 0.0775:
                        
                            sClassListString += "20\t" # classified into class 20   
                            
                        else:
                        
                            sClassListString += "21\t" # classified into class 21   
                        
                    elif fBodySize < 0.0809:
                        
                        if fEgg < 0.0370:
                        
                            sClassListString += "22\t" # classified into class 22   
                            
                        elif fEgg < 0.0775:
                        
                            sClassListString += "23\t" # classified into class 23   
                            
                        else:
                        
                            sClassListString += "24\t" # classified into class 24
                            
                    else:
                    
                        if fEgg < 0.0370:
                        
                            sClassListString += "25\t" # classified into class 25    
                            
                        elif fEgg < 0.0775:
                        
                            sClassListString += "26\t" # classified into class 26    
                            
                        else:
                        
                            sClassListString += "27\t" # classified into class 27    
                                   
            else:
                
                fLifeSpan = fLifeSpanList[nFileNum]
                
                if fLifeSpan < 9.3:
                    
                    # training set
                    if nFileNum not in nRandList:
                        
                        sOutFile12.write(sClassListString[:-1] + "\n")
                        
                    # test set
                    else:
                        
                        sOutFile13.write(sClassListString[:-1] + "\n")
                    
                elif fLifeSpan < 11.1:
                    
                    # training set
                    if nFileNum not in nRandList:
                        
                        sOutFile22.write(sClassListString[:-1] + "\n")
                        
                    # test set
                    else:
                        
                        sOutFile23.write(sClassListString[:-1] + "\n")
                    
                else:
                    
                    # training set
                    if nFileNum not in nRandList:
                        
                        sOutFile32.write(sClassListString[:-1] + "\n")
                        
                    # test set
                    else:
                        
                        sOutFile33.write(sClassListString[:-1] + "\n")
                    
                sInFile.close()
                
                nFileNum = nFileNum + 1
                
                break
        
sOutFile12.close()
sOutFile13.close()
sOutFile22.close()
sOutFile23.close()
sOutFile32.close()
sOutFile33.close()
