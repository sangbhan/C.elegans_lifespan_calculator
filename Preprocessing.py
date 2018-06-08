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

np.random.seed(777)

sPath = "C:/Users/Sangbin/Desktop/mmc2/processed/"

sFileList = os.listdir(sPath)

fMovementList = []
fBodySizeList = []
fEggList = []
fLifeSpanList = []

fEarlyMovementList = []
fEarlyBodySizeList = []
fEarlyEggList = []

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

plt.xlabel('Adult Lifespan ($days$)')
plt.ylabel('Number of Individuals')
plt.hist(np.array(fLifeSpanList), bins = [2, 4, 6, 8, 10, 12, 14, 16], facecolor = 'C2')
plt.title("Cohorts Across Disribution of Adult Lifespans")
plt.show()

plt.xlabel('Smarts')
plt.ylabel('Displacement Over 3 Hours ($mm$)')
plt.hist(np.array(fEarlyMovementList), bins = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], facecolor = 'C4')
plt.title("Cohorts Across Disribution of Early-Adulthood Movement")
plt.show()

plt.xlabel('Smarts')
plt.ylabel('Cumulative Area of Eggs Laid ($mm^2$)')
plt.hist(np.array(fEarlyEggList), bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], facecolor = 'C9')
plt.title("Cohorts Across Disribution of Early-Adulthood Cumulative Area of Eggs Laid")
plt.show()

plt.xlabel('Cross-Sectional Size (adjusted for machine bias) ($mm^2$)')
plt.ylabel('Number of Individuals')
plt.hist(np.array(fEarlyBodySizeList), bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11], facecolor = 'C3')
plt.title("Cohorts Across Disribution of Early-Adulthood Body Size")
plt.show()

sOutFile11 = open(sPath + "../../Data/short_lived_train.txt", "w")
sOutFile12 = open(sPath + "../../Data/short_lived_train_accuracy.txt", "w") # for calcluating training accuracy
sOutFile13 = open(sPath + "../../Data/short_lived_test_accuracy.txt", "w") # for calculating test accuracy

sOutFile21 = open(sPath + "../../Data/normal_lived_train.txt", "w")
sOutFile22 = open(sPath + "../../Data/normal_lived_train_accuracy.txt", "w") # for calcluating training accuracy
sOutFile23 = open(sPath + "../../Data/normal_lived_test_accuracy.txt", "w") # for calculating test accuracy

sOutFile31 = open(sPath + "../../Data/long_lived_train.txt", "w")
sOutFile32 = open(sPath + "../../Data/long_lived_train_accuracy.txt", "w") # for calcluating training accuracy
sOutFile33 = open(sPath + "../../Data/long_lived_test_accuracy.txt", "w") # for calculating test accuracy

nFileNum = 0

for sFile in sFileList:
    
    if sFile != 'metadata.tsv':
        
        sInFile = open(sPath + sFile, "r")
        sInFile.readline()
        
        sClassListString = ""
        
        fRandU = np.random.uniform(size = 1)
        
        for sReadLine in sInFile.readlines():
            
            sReadList = sReadLine.replace("\n", "").split("\t")
                            
            if float(sReadList[0]) <= 2:
                
                fMovement =  float(sReadList[22])
                fBodySize = float(sReadList[25])
                fEgg = float(sReadList[29])
                
                if fMovement < 0.4:
                    
                    if fBodySize < 0.06:
                        
                        if fEgg < 0.05:
                            
                            sClassListString += "1\t"
                            
                        elif fEgg < 0.2:
                            
                            sClassListString += "2\t"
                            
                        else:
                            
                            sClassListString += "3\t"
                        
                    elif fBodySize < 0.09:
                        
                        if fEgg < 0.05:
                            
                            sClassListString += "4\t"
                            
                        elif fEgg < 0.2:
                            
                            sClassListString += "5\t"
                            
                        else:
                            
                            sClassListString += "6\t"
                            
                    else:
                    
                        if fEgg < 0.05:
                            
                            sClassListString += "7\t"
                            
                        elif fEgg < 0.2:
                            
                            sClassListString += "8\t"
                            
                        else:
                            
                            sClassListString += "9\t"
                            
                elif fMovement < 0.6:
                    
                    if fBodySize < 0.06:
                        
                        if fEgg < 0.05:
                        
                            sClassListString += "10\t"    
                        
                        elif fEgg < 0.2:
                        
                            sClassListString += "11\t"    
                                                    
                        else:
                        
                            sClassListString += "12\t"    
                                                
                    elif fBodySize < 0.09:
                        
                        if fEgg < 0.05:
                        
                            sClassListString += "13\t"    
                            
                        elif fEgg < 0.2:
                        
                            sClassListString += "14\t"    
                            
                        else:
                        
                            sClassListString += "15\t"    
                            
                    else:
                    
                        if fEgg < 0.05:
                        
                            sClassListString += "16\t"    
                            
                        elif fEgg < 0.2:
                        
                            sClassListString += "17\t"    
                            
                        else:
                        
                            sClassListString += "18\t"    
                            
                else:
                    
                    if fBodySize < 0.06:
                        
                        if fEgg < 0.05:
                        
                            sClassListString += "19\t"    
                            
                        elif fEgg < 0.2:
                        
                            sClassListString += "20\t"    
                            
                        else:
                        
                            sClassListString += "21\t"    
                        
                    elif fBodySize < 0.09:
                        
                        if fEgg < 0.05:
                        
                            sClassListString += "22\t"    
                            
                        elif fEgg < 0.2:
                        
                            sClassListString += "23\t"    
                            
                        else:
                        
                            sClassListString += "24\t"    
                            
                    else:
                    
                        if fEgg < 0.05:
                        
                            sClassListString += "25\t"    
                            
                        elif fEgg < 0.2:
                        
                            sClassListString += "26\t"    
                            
                        else:
                        
                            sClassListString += "27\t"    
                                   
            else:
                
                fLifeSpan = fLifeSpanList[nFileNum]
                
                if fLifeSpan < 8:
                    
                    # training set
                    if fRandU < 0.5:
                        
                        sOutFile11.write(sClassListString + "0\t")
                        sOutFile12.write(sClassListString[:-1] + "\n")
                        
                    # test set
                    else:
                        
                        sOutFile13.write(sClassListString[:-1] + "\n")
                    
                elif fLifeSpan < 12:
                    
                    # training set
                    if fRandU < 0.5:
                        
                        sOutFile21.write(sClassListString + "0\t")
                        sOutFile22.write(sClassListString[:-1] + "\n")
                        
                    # test set
                    else:
                        
                        sOutFile23.write(sClassListString[:-1] + "\n")
                    
                else:
                    
                    # training set
                    if fRandU < 0.5:
                        
                        sOutFile31.write(sClassListString + "0\t")
                        sOutFile32.write(sClassListString[:-1] + "\n")
                        
                    # test set
                    else:
                        
                        sOutFile33.write(sClassListString[:-1] + "\n")
                    
                sInFile.close()
                
                nFileNum = nFileNum + 1
                
                break
        
sOutFile11.close()
sOutFile12.close()
sOutFile13.close()
sOutFile21.close()
sOutFile22.close()
sOutFile23.close()
sOutFile31.close()
sOutFile32.close()
sOutFile33.close()