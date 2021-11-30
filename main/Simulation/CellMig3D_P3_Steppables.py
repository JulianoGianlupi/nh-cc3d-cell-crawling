#########################################################################################################################################
# This code implements a simple model of a 3D crawling fibrobast-like cell with a nucleus (NUCL), cytoplasm (CYTO) and lamellipodium (FRONT).
# The model for the cell uses a single diffusing chemical field (FActin) confined to the cell volume which represents the region of actin polarization in the cell.
# The chemical field obeys equation 10 in the text: \partial FActin / \partial t = D_F \nabla^2 FActin + k_source (if inside lamellipodium) - k_decay * FActin.
# The chemical field applies a repulsive chemotactic force of strength \lambda_F-actin (equation 9) to the boundary between lamellipodium and medium only.
# The cell starts migrating spontaneously after its shape symmetry is broken. 
# The trajectory of the cell is calculated and recorded, along with some information about the symmetry breaking state of the cell.
# The code was written between 2014-2015 mainly by I. Fortuna and Gilberto L. Thomas of the IFUFRGS, Brazil.
# The work is published in Biophysical Journal 118, 2801–2815, June 2, 2020 (https://doi.org/10.1016/j.bpj.2020.04.024)
#########################################################################################################################################

"""2020-11-30 bugfixing by Juliano Ferrari Gianlupi. Fixes are indicated by a comment"""

#
# Generic Python housekeeping steps -- loads utility functions
#
import sys
import os
from cc3d.core.PySteppables import *
#import CompuCell
#import CompuCellSetup
#from PlayerPython import *
from numpy import *
from math import *
from random import *
from copy import deepcopy
import time
from os import makedirs
import os.path
import shutil
import inspect
from cc3d.core.XMLUtils import dictionaryToMapStrStr as d2mss
#
# Defines the Cell steppable class, which creates a cell object of type CYTO and its NUCL and CYTO and FRONT compartments.

class Cell(SteppableBasePy):
    #
    # Create the steppable and pass parameter values to it.
    #
    def __init__(self,frequency,phiF,phiN,cellRad,cellVol,x1,y1):
        SteppableBasePy.__init__(self,frequency) # Loads generic steppable class which executes at intervals specified by the CellMig3D_P3.py file in mcs (here, once per mcs).
        #self.nTrackerPlugin=CompuCell.getNeighborTrackerPlugin() # Allows the simulation to keep track of the identity of the cells and compartments.
        self.phiN=phiN;  self.phiF=phiF  # Sets fraction of cell volume allocated to NUCL and FRONT compartment as specified by the CellMig3D_P3.py file.
        self.phiC=1. - self.phiN - self.phiF  # Sets fraction of cell volume allocated to CYTO compartment as specified by the CellMig3D_P3.py file. 
        self.cellRad=cellRad;     self.cellVol=cellVol  # Sets cell radius and volume as specified by the CellMig3D_P3.py file.
        self.x1=x1;    self.y1=y1 # Sets initial cell center-of-mass positions in the x and y directions as specified by the CellMig3D_P3.py file.
    #
    # The "start" function is run once at the beginning of a simulation.
    # It generates the initial cell lattice configuration with a single cell object of type CYTO with two compartments (NUCL and CYTO).
    def start(self): 
        #
        # The following defines the various generalized cell objects and compartments.
        # For each object, we first need to create an object of this type in the cellList, then assign it a cell type, then assin voxels in the cell lattice to it.
        #
        seed(1000) # Initializes the python pseudo number generator to be reproductible
        pt=CompuCell.Point3D() # Creates a vector variable to hold a position (x,y,z). 
        SUBS_Acell=self.potts.createCell() # Creates a generalized cell object SUBS_Acell in the cellList list, which will be defined to be the rigid adhesive substrate at z=0.
                                           # Note that this method does NOT assign any voxels in the cell lattice to the cell object.
        SUBS_Acell.type=self.SUBS_A # Assigns the type of the generalized cell object SUBS_Acell to be SUBS_A, which has been defined in the Plug-ins in the CellMig3D_P3.py file.
        SUBS_NAcell=self.potts.createCell() # Creates a generalized cell object SUBS_NAcell in the cellList list, which will be defined to be the rigid nonadhesive substrate at z=z_max.
                                            # Note that this method does NOT assign any voxels in the cell lattice to the cell object.
        SUBS_NAcell.type=self.SUBS_NA # Assigns the type of the generalized cell object SUBS_NAcell to be SUBS_NA, which has been defined in the Plug-ins in the CellMig3D_P3.py file.
        Rad=int(self.cellRad) # Assigns the interger part of the cell radius defined in the CellMig3D_P3.py file to Rad.
        nCell=1 # Asserts that only one cell will be created.
        cell1=self.potts.createCell() # Creates a generalized cell object cell1 which will be defined to be the CYTO compartment of the cell.
                                      # Note that this method does NOT assign any voxels in the cell lattice to the cell object.
        cell1.type=self.CYTO # Assigns the type of the generalized cell object cell1 to be CYTO, which has been defined in the Plug-ins in the CellMig3D_P3.py file.
        x1CM=self.x1 # Sets the initial x-axis center-of-mass position of the cell object.
        y1CM=self.y1 # Sets the initial y-axis center-of-mass position of the cell object.
        z1CM=Rad # Sets initial cell center-of-mass positions in the z direction, so cell is tangent to the z=0 plane.
        #
        # The following loop iterates over every voxel in the cell lattice and assigns the appropriate voxels to the various generalized cell objects and compartments.
        # Initially the cell object of type CYTO has a CYTO compartment and a NUCL compartment but no FRONT compartment--the FRONT compartment forms. 
        # as a result of interactions with the adhesive substrate generalized cell object.
        #
        for x,y,z in self.every_pixel():
            pt.x=x; pt.y=y; pt.z=z # Puts the current x, y and z coordinates into the pt vector variable.
            if   ( pt.z==0 ): # If we are at the z=0 edge of the cell lattice universe we assign the voxel to form part of a layer of rigid adhesive substrate.
                self.cellField.set(pt,SUBS_Acell)  # Assign the current voxel to the rigid adhesive substrate generalized cell object.
            elif ( pt.z==self.dim.z-1 ): # If we are at the z=z_max edge of the cell lattice universe we assign the voxel to form part of a layer of rigid nonadhesive substrate.
                self.cellField.set(pt,SUBS_NAcell) # Assign the current voxel to the rigid nonadhesive substrate generalized cell object.
            elif ( sqrt((x-x1CM)*(x-x1CM)+(y-y1CM)*(y-y1CM)+(z-z1CM)*(z-z1CM)) < Rad): # If we are within a sphere of radius Rad around the initial center-of-mass coordinates of the 
                                                                                       # cell object we assign the voxel to form part of the CYTO compartment of the cell object.
                self.cellField.set(pt,cell1) # Assign the current voxel to the CYTO compartment of the cell object of type CYTO.
            # Any voxels not selected in the if statement are left in the default generalized cell index 0 of generalized cell type "Medium".   
            #
            # Now we reassign some of the voxels in the cell object of type CYTO to be part of the NUCL compartment.
            # initially we have no FRONT compartment--it forms as a result of interactions with the adhesive substrate.
            # 
        for cell in self.cellListByType(self.CYTO):  # We now iterate over all cell objects of type CYTO. In this simulation we have only one such object.
            print( "******************************",cell.id,len(self.cellListByType(self.CYTO)))
            pt.x=int(cell.xCOM);  pt.y=int(cell.yCOM);  pt.z=int(cell.zCOM) # We determine the current center-of-mass position of the current CYTO compartment in the cell object.
            NUCLcell=self.potts.createCellG(pt) # Creates a generalized cell object NUCLcell which will later be defined to be the NUCL compartment of the cell of type CYTO.
                                                                  # Note that this method does NOT assign any voxels in the cell lattice to the cell object.
            NUCLcell.type=self.NUCL # Assigns the type of the generalized cell object NUCLcell to be NUCL, which has been defined as a compartment in the Plug-ins in the CellMig3D_P3.py file. 
            # 
            # The following 3 loops assigns a small square (7 x 7 x 7) of voxels around the cell's center of mass to the NUCL generalized cell object.
            # The NUCL cell object  will rapidly round up and adjust its volume to be that specified in the CellMig3D_P3.py file .
            #
            for i in range(-3,3):
                for ii in range(-3,3):
                    for iii in range(-3,3):
                        pt.x=int(cell.xCOM)+i; pt.y=int(cell.yCOM)+ii; pt.z=int(cell.zCOM)+iii
                        self.cellField.set(pt,NUCLcell) # Assigns the current voxel to the NUCLcell generalized cell object.
            #
            # Sets the nucleus (NUCLcell) generalized cell object target volume and inverse compressibility (see equation 8).
            #
            NUCLcell.targetVolume=int(self.cellVol*self.phiN)+.5 
            NUCLcell.lambdaVolume=10
            #
            # Updates the cytoplasm (CYTO) compartment target volume and inverse compressibility (see equation 8) to reflect the assignment of voxels to the nucleus.
            #
            cell.targetVolume=int(self.cellVol*(self.phiC+self.phiF))+.5
            cell.lambdaVolume=10
            print( "@@@@@@@@@@@@@@@",NUCLcell.id,cell.clusterId)
            reassignIdFlag=self.inventory.reassignClusterId(NUCLcell,cell.clusterId)  # Make the NUCLcell generalized cell object a compartment of the current cell.
    #
    # The "step" function runs once per mcs.
    # It generates the FRONT compartment in the cell when the cell's CYTO compartment touches the adhesive substrate.
    #
    def step(self,mcs): 
        pt=CompuCell.Point3D() # Creates a vector variable to hold a position (x,y,z).
        #
        # The next loop creates the FRONT compartment from part of the CYTO compartment of the cell object of type CYTO.
        #
        for cell in self.cellListByType(self.CYTO):  # Identifies any cell(s) of type CYTO. 
            nFRONT=0;   FRONTvol=0.0;   NUCLvol=0.0 # Defines counters to measure the current volume of the three compartments.
            """JFG bugfix for 4.2.5"""
            # compList=self.inventory.getClusterCells(cell.clusterId) # Creates a list of the compartments in the cell.
            compList = self.get_cluster_cells(cell.clusterId)
            """/JFG bugfix for 4.2.5"""
            # for cell_cmpt in cluster_cell_list:
            #     print('compartmental cell id=', cell_cmpt)

            for compCell in compList: # Iterates over the compartments in the cell.
                if compCell.type==self.NUCL: # The following three lines assign the actual volume of the NUCL compartment to the variable NUCLvol.
                    NUCLcell=compCell
                    NUCLvol=compCell.targetVolume
                elif compCell.type==self.FRONT: # The following three lines sum over all compartments in the cell of type FRONT and accumulate the actual volume 
                                                # in FRONTvol and counts the number of FRONT compartments in the cell in nFRONT.
                    FRONTcell=compCell
                    FRONTvol+=compCell.targetVolume
                    nFRONT+=1
            CELLvol = cell.targetVolume + FRONTvol + NUCLvol # The total actual cell volume (Cellvol) is the sum of the CYTO, FRONT and NUCL compartment actual volumes.
            pFRONT = 0.1*(1.- FRONTvol/CELLvol/self.phiF) # pFRONT allows the gradual growth of the FRONT compartment until it reaches a fraction phiF of the total cell volume.
            if pFRONT>0:
                pList=[]; pixelList=self.get_cell_boundary_pixel_list(cell)  # Creates a list of all CYTO compartment boundary voxels.
                for pixel in pixelList: # Iterates over the list of all CYTO compartment boundary voxels.
                    if pixel.pixel.z==1: # Considers only the CYTO compartment boundary voxels which also contact the adherent substrate along the z=0 boundarys.
                        pList.append([pixel.pixel.x,pixel.pixel.y,pixel.pixel.z]) # pList accumulates all CYTO compartment boundary voxels which also contact the adherent substrate along the z=0 boundary.
                shuffle(pList) # Randomizes the order of the CYTO compartment boundary voxels which also contact the adherent substrate along the z=0 boundary.
                for pixel in range(len(pList)): # Iterates through the shuffled list of CYTO compartment boundary voxels which also contact the adherent substrate along the z=0 boundary.
                    if (random()<pFRONT): # Determines whether to reassign the current CYTO voxel to the FRONT compartment with probability pFRONT.
                        pt.x=pList[pixel][0]; pt.y=pList[pixel][1]; pt.z=pList[pixel][2]  # Determines the coordinates of the current CYTO voxel to change to a FRONT voxel.
                        if not nFRONT: # If the cell does not currently have a FRONT compartment.
                            FRONTcell=self.potts.createCell() # Creates a generalized cell object FRONTcell which will be defined to be a compartment of the cell of type FRONT.
                                                              # Note that this method does NOT assign any voxels in the cell lattice to the cell object.
                            FRONTcell.type=self.FRONT # Assigns the type of the FRONTcell generalized cell object to be FRONT.
                            self.cellField.set(pt,FRONTcell)   # Assigns the current voxel to be part of the FRONTcell generalized cell object.
                            reassignIdFlag=self.inventory.reassignClusterId(FRONTcell,cell.clusterId)  # Makes the FRONTcell generalized cell object a compartment of the cell.
                            FRONTcell.targetVolume=1.5 # Assigns the initial target volume of the FRONT cell compartment to be 1.5 voxels (see equation 8).
                            FRONTcell.lambdaVolume=10 # Assigns the inverse compressibility of the FRONT cell compartment (see equation 8).
                            cell.targetVolume-=1.5 # Decreases the volume of the cell by the amount of volume transferred to the FRONT cell compartment.
                            nFRONT+=1 # Increase the number of FRONT compartments in the cell by one.
                            FRONTvol=1.5
                        else:  # If the cell already has a FRONT compartment.
                            self.cellField.set(pt,FRONTcell) # # Assigns the current voxel to be part of the current FRONT compartment.
                            FRONTcell.targetVolume+=1.  # Increases the target volume of the FRONT cell compartment by 1.5 voxels (see equation 8).
                            cell.targetVolume-=1. # Decreases the volume of the cell by the amount of volume transferred to the FRONT cell compartment.
                            FRONTvol+=1.
                        pFRONT = 0.1*(1.- FRONTvol/CELLvol/self.phiF)
#
# Defines the Calc steppable class, which measures the properties of the migrating cell.
#

class Calc(SteppableBasePy):
    #
    # The following defines the output files, the plots to be displayed, and the preliminary analysis of the cell's properties and movements 
    #
    def __init__(self,frequency,phiF,phiN,lambCHEM,cellRad,g1Flag,g2Flag,x1,y1,dT): 
        SteppableBasePy.__init__(self,frequency) # Loads generic steppable class which executes at intervals specified by the CellMig3D_P3.py file in mcs 
                                                                                         # (here, once per deltaT= 50 mcs).
        #self.nTrackerPlugin=CompuCell.getNeighborTrackerPlugin() # Allows the simulation to keep track of the identity of the cells and compartments.
        self.phiF=phiF;    self.phiN=phiN # Sets fraction of cell volume allocated to NUCL and FRONT compartment as specified by the CellMig3D_P3.py file.
        self.phiC=1. - self.phiF - self.phiN # Sets fraction of cell volume allocated to CYTO compartment as specified by the CellMig3D_P3.py file.
        self.lambCHEM=lambCHEM # Sets the strength of the chemotaxis response in equation 9 as specified by the CellMig3D_P3.py file .
        self.deltaT=dT # Sets the interval between measurements to deltaT=50 mcs as specified by the CellMig3D_P3.py file. 
        self.cellRad=cellRad # Sets cell the radius as specified by the CellMig3D_P3.py file. 
        self.g1Flag=g1Flag;  self.g2Flag=g2Flag # Turns on the display of graphics in Player as specified by the CellMig3D_P3.py file. 
        self.x1=x1;    self.y1=y1 # Sets the initial x and y center-of-mass positions of the cell .object as specified by the CellMig3D_P3.py file.
    #
    # The "start" function is run once at the beginning of a simulation.
    # It generates the output files and plot windows.
    #
    
    def start(self):
        #
        # The following lines build the output filenames from the values of the parameters used in the simulation. This name provides an easy way to identify the data the simulations produce.
        # There are two output files. They have the same prefix containing the parameter information: one for the symmetry-breaking data,  the other for the centers-of-mass
        # of the compartments, both as function of time (mcs) writen in the first column.
        #
        filePath = os.path.abspath(__file__)
        
        
#         Dir=CompuCellSetup.getScreenshotDirectoryName() # Defines the output directory for the files.
        Dir=filePath # Defines the output directory for the files.
        output="_R"+str(self.cellRad) + "_pF" + str(self.phiF) + "_lC" + str(self.lambCHEM) + "_dT" + str(self.deltaT) #  Defines the output file name prefix based on the parameters.
        #
        self.SB=output + "_SBAn.dat" # Creates full file name for symmetry breaking data, as explained in in the Supplementary Materials.
        self.brokenSB="_BROKEN" + output + "_SBAn.dat" # Creates full file name for symmetry breaking data for simulations in which the FRONT compartment separates from the CYTO compartment.
        #
        self.SBfileName=Dir + self.SB # Creates full path name for symmetry breaking data.
        self.brokenSBfileName=Dir + self.brokenSB # Create full path name for symmetry breaking data for simulations in which the FRONT compartment separates from the CYTO compartment.  
        SBfile=open(self.SBfileName,'w') # Creates the file for symmetry breaking data for write access.
        #
        # The next line writes an alphanumeric header as the first line in the symmetry breaking data file. It identifies the columns in it, as explained in the Supplementary Materials.
        #
        SBfile.write("%s\n" % ("time    dcm_F_CN    zposN    Cont_CF     phi3d_C    phi3d_F    phi3d_N    cellVol"))
        # Columns are time in mcs, distance between the center of mass of the FRONT compartment and the combined center of mass of the CYTO and NUCL compartments, 
        # the z position of the center of mass of the NUCL compartment, area of boundary between the CYTO and FRONT compartments, volume fraction of CYTO compartment, 
        # volume fraction of FRONT compartment, volume fraction of NUCL compartment, volume of the cell.
        SBfile.close() # Forces storage of the information by closing the file.
        #
        self.CD=output + "_Displacement.dat" # Creates full file name for centers-of-mass data on the compartments, as explained in the Supplementary Materials.
        self.brokenCD="_BROKEN" + output + "_Displacement.dat" # Creates full file name for centers-of-mass data for simulations in which the FRONT compartment separates from the CYTO compartment.
        #
        self.CDfileName=Dir + self.CD # Creates full path name for centers-of-mass data.
        self.brokenCDfileName=Dir + self.brokenCD # Creates full path name for centers-of-mass data for simulations in which the FRONT compartment separates from the CYTO compartment.
        CDfile=open(self.CDfileName,'w') # Creates the file for centers-of-mass data for write access.
        CDfile.write("%s\n" % ("time   xposC       yposC      zposC       xposF      yposF     zposF       xposN       yposN      zposN       xposCN      yposCN    zposCN"))
        # Columns are time in mcs, x center of mass of CYTO compartment, y center of mass of CYTO compartment, z center of mass of CYTO compartment,
        # x center of mass of FRONT compartment, y center of mass of FRONT compartment, z center of mass of FRONT compartment,
        # x center of mass of NUCL compartment, y center of mass of NUCL compartment, z center of mass of NUCL compartment, and
        # joint x center of mass of CYTO plus NUCL compartment, joint y center of mass of CYTO plus NUCL compartment, joint z center of mass of CYTO plus NUCL compartment.
        CDfile.close() # Forces storage of the information by closing the file.
        #
        # Prints file names and path to console for reference.
        #
        print( "**********************")
        print( Dir) # Prints path to data files
        print( self.SBfileName )# Prints name of symmetry breaking data file.
        print( self.CDfileName )# Prints name of centers-of-mass data file.
        print( "**********************")
        #
        # The next lines create the plot windows for the calculated data. Please refer to CC3D documentation for comprehensive discussion of function arguments.
        #
        if self.g1Flag=='yes': # If the flag for plotting displacements is set.
            #
            # CELL DISPLACEMENT
            # plotWindow 1 [NUCL trajectory]
            #
            self.pW1=self.addNewPlotWindow(_title='Nucleus Trajectory',_xAxisTitle='x (pixel)',_yAxisTitle='y (pixel)', _xScaleType='linePlotWindowInterfacear',_yScaleType='linear') # Opens and names plot window.
#            self.pW1.addPlot("trajN",_style='Dots',_color='red',_size=1)   # Defines color of (0,0) reference point.
            self.pW1.addPlot("trajN",_style='Lines',_color='red',_size=1)   # Defines color of (0,0) reference point.
            self.pW1.addPlot("trajScale",_style='Dots',_color='black')   # Hides max and min reference points.
            # self.pW1.add_data_point("trajN",0,0) # Puts a point at the 0,0 position on the graph.
            # self.pW1.add_data_point("trajScale",-2.*self.dim.x/3/self.cellRad,-2.*self.dim.y/3/self.cellRad) # Puts a point at the minimum position to set graph axis scales.
            # self.pW1.add_data_point("trajScale",2.*self.dim.x/3/self.cellRad,2.*self.dim.y/3/self.cellRad) # Puts a point at the maximum position to set graph axis scales.
            # self.pW1.show_all_plots() # Displays plot to Player.
#             self.pW1.addAutoLegend("top") # Puts a legend at the top of the graph window.
        #
        if self.g2Flag=='yes': # If the flag for plotting cell properties is set.
            #
            # SYMMETRY BREAK
            #
            # plotWindow 2 [volume fractions of the three compartments].
            #
            self.pW2=self.addNewPlotWindow(_title='3d cell fraction',_xAxisTitle='time [mcs]',_yAxisTitle='3d cell fraction', _xScaleType='linear',_yScaleType='linear')
            self.pW2.addPlot("phiC",_style='Lines',_color='blue',_size=2)      
            self.pW2.addPlot("phiF",_style='Lines',_color='green',_size=2)    
            self.pW2.addPlot("phiN",_style='Lines',_color='yellow',_size=2)   
#             self.pW2.addAutoLegend("top")
            #
            # plotWindow 3 [distance between FRONT and combined CYTO+NUCL combined centers-of-mass].
            #
            self.pW3=self.addNewPlotWindow(_title='Distance F-CN',_xAxisTitle='time [mcs]',_yAxisTitle='CM distances', _xScaleType='linear',_yScaleType='linear')
            self.pW3.addPlot("dCM",_style='Lines',_color='red')   
#             self.pW3.addAutoLegend("no")
            #
            # plotWindow 4 [NUCL z position].
            #
            self.pW4=self.addNewPlotWindow(_title='Nucleus z position',_xAxisTitle='time [mcs]',_yAxisTitle='z(CM)', _xScaleType='linear',_yScaleType='linear')
            self.pW4.addPlot("zposN",_style='Lines',_color='red')   
            #
            # plotWindow 5 [Contact amount between CYTO - FRONT].
            #
            self.pW5=self.addNewPlotWindow(_title='CYTO-FRONT Contact',_xAxisTitle='time [mcs]',_yAxisTitle='# of voxels', _xScaleType='linear',_yScaleType='linear')
            self.pW5.addPlot("contCF",_style='Lines',_color='red')   
        #
        # Now starts the cell and compartment centers-of-mass calculations. 
        # Creates and zeros vectors to store the (X,Y,Z) coordinates at time t and at time t+dTn and the displacement during this interval.
        # Vector names: Let "..." represent the compartments (C,F,N). 
        # Then, self.opos... -> coordinates at t; self.pos... -> coordinates at t+dT; and self.desl... -> displacement between t and t+dT.
        #
        self.oposC=[.0,.0,.0];   self.posC=[.0,.0,self.cellRad];    self.deslC=[.0,.0,.0]
        self.oposF=[.0,.0,.0];   self.posF=[.0,.0,self.cellRad];    self.deslF=[.0,.0,.0]
        self.oposN=[.0,.0,.0];   self.posN=[.0,.0,self.cellRad];    self.deslN=[.0,.0,.0]
        #
        # For each cell in the cell list (only one in this simulation) store the center of mass position as a placeholder at t=0 to compare to later positions of each compartment.
        #
        for cell in self.cellListByType(self.CYTO): # Loop over all cells
            self.oposC=[cell.xCOM,cell.yCOM,cell.zCOM] # Store the cell center of mass as a placeholder at t=0 to compare to later positions of the CYTO compartment.
            self.oposF=[cell.xCOM,cell.yCOM,cell.zCOM] # Store the cell center of mass as a placeholder at t=0 to compare to later positions of the FRONT compartment.
            self.oposN=[cell.xCOM,cell.yCOM,cell.zCOM] # Store the cell center of mass as a placeholder at t=0 to compare to later positions of the NUCL compartment.          
        self.phiTarget=[self.phiC,self.phiF,self.phiN] # Here we store the volume fraction of each compartment with respect to the whole call.
    #
    # The "step" function runs once per mcs.
    # It analyzes the cell shape and positions and outputs the results to Player plots and to the output files once every deltaT=50 mcs.
    #
    def step(self,mcs):  # The step function is called every mcs.
        pt=CompuCell.Point3D() # Creates a vector variable to hold a position (x,y,z).
        # 
        # Cell movement analysis.
        #
        if (mcs%self.deltaT==0): # The calculations occur every deltaT mcs.
            self.phi3d=[.0,.0,.0] # Creates and zeros a vector to store [phiCYTO,phiFRONT,phiNUCL]; fraction of compartment volumes.
            self.Cont_CF=.0 # Creates and zeros a variable to count the number of CYTO-FRONT contact voxels.
            #
            for cell in self.cellListByType(self.CYTO): # Loop over all cells        
                self.deslC[0]=cell.xCOM-self.oposC[0] # The next 3 lines calculate the displacement of the current CYTO compartment center-of-mass position from the its previous position.
                self.deslC[1]=cell.yCOM-self.oposC[1]            
                self.deslC[2]=cell.zCOM-self.oposC[2]
                self.oposC[0]=cell.xCOM;  self.oposC[1]=cell.yCOM;  self.oposC[2]=cell.zCOM  # Stores the current CYTO center of mass position as the previous position.
                if   self.deslC[0]<-self.dim.x/2.: self.deslC[0]+=self.dim.x # The next 4 lines adjust the displacements for the situation where the CYTO center of mass crosses one of the xy periodic boundaries.
                elif self.deslC[0]> self.dim.x/2.: self.deslC[0]-=self.dim.x 
                if   self.deslC[1]<-self.dim.y/2.: self.deslC[1]+=self.dim.y 
                elif self.deslC[1]> self.dim.y/2.: self.deslC[1]-=self.dim.y 
                #
                self.posC[0]+=self.deslC[0] # The next 3 lines accumulate the total displacement of the CYTO compartment as if the cell object were crawling in an infinte lattice rather than a periodic lattice.
                self.posC[1]+=self.deslC[1] 
                self.posC[2]+=self.deslC[2] 
                self.phi3d[0]+=cell.volume  # Accumulates the cell object volume in phi3d[0].
                #cellNeighborList = CellNeighborListAuto(self.nTrackerPlugin,cell)  # Creates a list of the generalized cell and compartment objects adjacent to the cell object.
                cellNeighborList = self.get_cell_neighbor_data_list(cell)
                for neighbor, common_surface_area in cellNeighborList: # Iterate over the neighbors of the cell object.
                    if neighbor: # If the neighbor object is not "Medium".
                        Type2name=neighbor.type # Store the type of the neighbor object.
                        if (Type2name==self.FRONT):   self.Cont_CF+=common_surface_area # Accumulate the contact area between the cell object and the FRONT compartment.
                # 
                # Calculates the FRONT (X,Y,Z) displacement, in the same way as above (lines 319-332).
                #
                """JFG bugfix for 4.2.5"""
                # compList=self.inventory.getClusterCells(cell.clusterId) # Stores list of the compartments of the cell object.
                compList = self.get_cluster_cells(cell.clusterId)
                """/JFG bugfix for 4.2.5"""
                for compCell in compList:               
                    if  compCell.type==self.FRONT:             
                        self.deslF[0]=compCell.xCOM-self.oposF[0]
                        self.deslF[1]=compCell.yCOM-self.oposF[1]
                        self.deslF[2]=compCell.zCOM-self.oposF[2]
                        self.oposF[0]=compCell.xCOM;                self.oposF[1]=compCell.yCOM;                 self.oposF[2]=compCell.zCOM
                        if   self.deslF[0]<-self.dim.x/2.: self.deslF[0]+=self.dim.x
                        elif self.deslF[0]> self.dim.x/2.: self.deslF[0]-=self.dim.x
                        if   self.deslF[1]<-self.dim.y/2.: self.deslF[1]+=self.dim.y
                        elif self.deslF[1]> self.dim.y/2.: self.deslF[1]-=self.dim.y
                        self.posF[0]+=self.deslF[0]
                        self.posF[1]+=self.deslF[1]
                        self.posF[2]+=self.deslF[2]
                        self.phi3d[1]+=compCell.volume
                    elif compCell.type==self.NUCL:       # Here it calculates the NUCL displacement, in the same way as above (lines 319-332).
                        self.deslN[0]=compCell.xCOM-self.oposN[0]
                        self.deslN[1]=compCell.yCOM-self.oposN[1]
                        self.deslN[2]=compCell.zCOM-self.oposN[2]
                        self.oposN[0]=compCell.xCOM;                self.oposN[1]=compCell.yCOM;                 self.oposN[2]=compCell.zCOM
                        if   self.deslN[0]<-self.dim.x/2.: self.deslN[0]+=self.dim.x
                        elif self.deslN[0]> self.dim.x/2.: self.deslN[0]-=self.dim.x
                        if   self.deslN[1]<-self.dim.y/2.: self.deslN[1]+=self.dim.y
                        elif self.deslN[1]> self.dim.y/2.: self.deslN[1]-=self.dim.y
                        self.posN[0]+=self.deslN[0]
                        self.posN[1]+=self.deslN[1]
                        self.posN[2]+=self.deslN[2]
                        self.phi3d[2]+=compCell.volume
            #
            cellVol=sum(self.phi3d) # Sums all compartment volumes.
            posCN=[.0,.0,.0] # Creates and zeros a vector to record the position of CYTO+NUCL (CN) combined center-of mass.
            d_F_CN=.0 # Creates and zeros a variable to store the distance between the centers-of-mass of FRONT and (CYTO+NUCL).
            for i in range(3): # Calculates the distance between F and CN
                posCN[i]= (self.posC[i]*self.phi3d[0]+self.posN[i]*self.phi3d[2])/(self.phi3d[0]+self.phi3d[2])
                if not i==2: d_F_CN+=(posCN[i]-self.posF[i])*(posCN[i]-self.posF[i])
            # Scales all the calculations by the cell size, CELLRad.
            xC=self.posC[0]/self.cellRad ; yC=self.posC[1]/self.cellRad; zC=self.posC[2]/self.cellRad
            xF=self.posF[0]/self.cellRad ; yF=self.posF[1]/self.cellRad; zF=self.posF[2]/self.cellRad
            xN=self.posN[0]/self.cellRad ; yN=self.posN[1]/self.cellRad; zN=self.posN[2]/self.cellRad
            xCN=posCN[0]/self.cellRad ; yCN=posCN[1]/self.cellRad; zCN=posCN[2]/self.cellRad
            d_F_CN=sqrt(d_F_CN)/self.cellRad # Calculate RMS displacement scaled by cell size.
            hN=self.posN[2]/self.cellRad # Calculate rescaled z position of NUCL compartment. 
            self.phi3d/=sum(self.phi3d) # Calculates the volume fraction for each compartment.
            #
            # Plot and store the data.
            #
            if (self.g1Flag=='yes'): # Display if the first display flag is set.
#                self.pW1.add_data_point("trajN",self.posN[0]/self.cellRad,self.posN[1]/self.cellRad) # Plots the xy-projection of the NUCL compartment.
                self.pW1.add_data_point("trajN",xN,yN) # Plots the xy-projection of the NUCL compartment.
#                self.pW1.add_data_point("trajN",0,0) # Puts a point at the 0,0 position on the graph.
                self.pW1.add_data_point("trajScale",-2.*self.dim.x/3/self.cellRad,-2.*self.dim.y/3/self.cellRad) # Puts a point at the minimum position to set graph axis scales.
                self.pW1.add_data_point("trajScale",2.*self.dim.x/3/self.cellRad,2.*self.dim.y/3/self.cellRad) # Puts a point at the maximum position to set graph axis scales.
                self.pW1.show_all_plots() # Update plots in Player
            #
            if (self.g2Flag=='yes'): # Display if the second display flag is set.
                self.pW2.add_data_point("phiC",mcs,self.phi3d[0]) # Plots the CYTO compartment volume fraction.
                self.pW2.add_data_point("phiF",mcs,self.phi3d[1]) # Plots the FRONT compartment volume fraction.
                self.pW2.add_data_point("phiN",mcs,self.phi3d[2]) # Plots the NUCL compartment volume fraction.
                self.pW2.show_all_plots() # Update plots in Player
                self.pW3.add_data_point("dCM",mcs,d_F_CN) # Plots the distance between the centers-of-mass of FRONT and (CYTO+NUCL).
                self.pW3.show_all_plots() # Update plots in Player.
                self.pW4.add_data_point("zposN",mcs,hN) # Plots the NUCL center-of-mass z-position.
                self.pW4.show_all_plots() # Update plots in Player.
                self.pW5.add_data_point("contCF",mcs,self.Cont_CF) # Plots the contact area of the boundary between the CYTO and FRONT compartments.
                self.pW5.show_all_plots()# Update plots in Player.
            #
            # Records output data to previously opened files--see above for meaning of columns.
            #
            SBfile=open(self.SBfileName,'a')  # Reopens file for symmetry breaking data for append.
            SBfile.write("%d %.8f %.8f %d %.8f %.8f %.8f %d  \n" % (mcs,d_F_CN,hN,self.Cont_CF,self.phi3d[0],self.phi3d[1],self.phi3d[2],cellVol)) # Store current symmetry breaking data to symmetry breaking data file
            SBfile.close()# Force storage of the information by closing the file.
            CDfile=open(self.CDfileName,'a')  # Reopens file for centers-of-mass data for append.
            CDfile.write("%d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f \n" % (mcs ,xC,yC,zC,xF,yF,zF,xN,yN,zN,xCN,yCN,zCN )) # Store current centers-of-mass data to centers-of-mass data file
            CDfile.close() # Force storage of the information by closing the file.   
            #
            # The lines below check if the cell's FRONT compartment has detached from the cell's CYTO compartment. If so, the simulation has failed and it renames the output 
            # files to identify the detachment and halts the simulation so that the simulation can resume with a different set of parameters.
            #
            if not self.Cont_CF and mcs > 10: # If the FRONT compartment is detached from the CYTO compartment of the cell and the time is greater than 10 mcs.
                os.system("move "+'"'+self.CDfileName+'"'+" "+'"'+self.brokenCDfileName+'"') # Renames the symmetry-breaking data file.
                os.system("move "+'"'+self.SBfileName+'"'+" "+'"'+self.brokenSBfileName+'"') # Renames the centers-of-mass data file.
                print( " -==// Simulation ending \\==-  mcs = ",mcs )# Prints a warning to the console that the simulation is terminating early due to detachment.
                self.stopSimulation() # End this instance of the simulation and continue with the next set of parameters.
    #
    # The "finish" function runs once after the simulation has completed its specified number of mcs.
    #
    def finish(self):
        pass # The finish function is null in this simulation.