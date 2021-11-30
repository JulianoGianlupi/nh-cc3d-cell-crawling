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
#
# import sys,time
# from os import environ
# from os import getcwd
from cc3d import CompuCellSetup
# import string
#
# Defines the names of the parameters and persistent variables to be used here and in the steppables 
#
global cellRad, cellVol # "global" allows these variables to be shared between Python modules (by default they are local to their module of specification)
global Lx, Ly, Lz # Parameters specifying the cell lattice size along x, y and z axes, respectively
global x1,y1,z1 # Parameters to set the initial center of mass position of the cell for x, y and z directions, respectively
global J0, T, lambCHEM # Parameters specifying the contact energy per voxel-voxel contact scale (equation 7), the modified  Metropolis dynamics fluctuation amplitude (paragraph before equation 6) and the strength (equation 9) by which the lamellipodia of cells respond to the FActin chemical field by chemotaxis
global phiN, phiF, phiC # Parameters specifying the fraction of cell volume allocated to nucleus, lamellipodium and cytoplasm respectively (see Table 2)
global tSim,deltaT # Parameters specifying the duration of the simulation in MonteCarlo steps (mcs) and the interval between measurements by the analysis tools in mcs
global g1Flag, g2Flag # Parameter flags specifying what information to display as graphical output in CC3D Player
global RANDOM_SEED # Parameter specifying the seed for the pseudo-random number generator in the stochastic components of the simulation
# use of identical seeds may not produce identical output because of differences between CC3D versions and differences between pseudo-random number generators on different computers and operating systems
#
# Define the typical linear size of the cell
cellRad=10.0                 # values used in ParameterScan are: 10.0, 15.0, 20.0, 30.0
cellVol=4.19*cellRad*cellRad*cellRad # Define the volume of the cell in terms of its radius
#
# Defines relative cell compartment volumes
#
phiF=0.05                    # Fraction of cell volume allocated to FRONT Compartment, FRONT % cell volume; values used in ParameterScan are: 0.05,0.10,0.20,0.30
phiN=0.15                    # Fraction of cell volume allocated to NUCL compartment, NUCL % cell volume
phiC=1.- phiN - phiF     # Fraction of cell volume allocated to CYTO compartment, CYTO % cell volume
lambCHEM=-150.        # stength \lambda_F-actin (in equation 9) of cell's FRONT compartment chemotaxis response to F-actin "field" FActin; values used in ParameterScan are:  0.,-50.,-100.,-150., -175., -200.
#
# Defines grid size and CPM parameters
#                                    
Lx=int(8*cellRad) # Sets the size of the x axis of the cell lattice to be 8 times the cell radius by default
if cellRad>=20. and phiF >= 0.2:
    Lx=int(10*cellRad) # Sets the size of the x axis of the cell lattice to be 10 times the cell radius if the cell radius is bigger than 20 and the fraction of the cell that is allowed to become lamellipodium is 20% or greater
if cellRad>=30. and phiF >= 0.2:
    Lx=int(12*cellRad) # Sets the size of the x axis of the cell lattice to be 12 times the cell radius if the cell radius is bigger than 30 and the fraction of the cell that is allowed to become lamellipodium is 20% or greater
if cellRad>=40. and phiF >= 0.2:
    Lx=int(14*cellRad) # Sets the size of the x axis of the cell lattice to be 14 times the cell radius if the cell radius is bigger than 40 and the fraction of the cell that is allowed to become lamellipodium is 20% or greater
Ly=Lx # Sets the size of the y axis of the cell lattice to be the same as the size of the x axis to define the lattice to be square in the x and y directions
Lz=int(2.1*cellRad) # sets the z-axis of the cell lattice to be just big enough to accomodate the cell so it does not hit the top (z_max) boundary in the z direction
x1=int(Lx/2.);  y1=int(Ly/2.) # Sets the initial x-y center-of-mass position of the cell to be the x-y center of the cell lattice
T=100.      # The simulation parameter T determines the amplitude of fluctuations in the Modified Metropolis dynamics, defined in the text body paragraph prior to equation 6
J0=20.      # Defines the energy scale for cell-cell adhesion per unit voxel-voxel contact, see equation 7 and below in the "Contact" and "ContactInternal" plug-ins
deltaT=50  # interval between measurements in mcs; =1 if velocity autocorrelation functions being calculated
tSim=100001         
RANDOM_SEED=1        # Provides a placeholder for the seed for the pseudo-random number generator from the ParameterScan file
#
# GRAPH plotting flags (yes/no): g1Flag - xy projection NUCL trajectory; g2Flag - others
#
g1Flag='yes' # Turn on display of graphs of cell movement properties
g2Flag='yes' # Turn on display of graphs of cell movement properties                         
#
#    CC3D's "configureSimulation" function defines the model structure to be implemented in the simulation 
#(Objects, Properties, Behaviors, Interactions, Dynamics, Initial Conditions and Boundary Conditions)
#
def configure_simulation():

    from cc3d.core.XMLUtils import ElementCC3D
    
    CompuCell3DElmnt=ElementCC3D("CompuCell3D",{"Revision":"20200724","Version":"4.2.2"})
    
    MetadataElmnt=CompuCell3DElmnt.ElementCC3D("Metadata")
    MetadataElmnt.ElementCC3D("NumberOfProcessors",{},"1")
    MetadataElmnt.ElementCC3D("DebugOutputFrequency",{},1000)
    
    PottsElmnt=CompuCell3DElmnt.ElementCC3D("Potts")
    PottsElmnt.ElementCC3D("Dimensions",{"x":Lx,"y":Ly,"z":Lz})
    PottsElmnt.ElementCC3D("Steps",{},tSim)
    PottsElmnt.ElementCC3D("Temperature",{},T)
    PottsElmnt.ElementCC3D("RandomSeed",{},RANDOM_SEED)
    PottsElmnt.ElementCC3D("NeighborOrder",{},"1")
    PottsElmnt.ElementCC3D("Boundary_x",{},"Periodic")
    PottsElmnt.ElementCC3D("Boundary_y",{},"Periodic")
    
    PluginElmnt=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"CellType"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"0","TypeName":"Medium"})
    PluginElmnt.ElementCC3D("CellType",{"Freeze":"","TypeId":"1","TypeName":"SUBS_A"})
    PluginElmnt.ElementCC3D("CellType",{"Freeze":"","TypeId":"2","TypeName":"SUBS_NA"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"3","TypeName":"CYTO"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"4","TypeName":"FRONT"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"5","TypeName":"NUCL"})
    
    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Volume"})
#    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Surface"})
    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"CenterOfMass"})
    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"NeighborTracker"})
    
    PluginElmnt_1=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"BoundaryPixelTracker"})
    PluginElmnt_1.ElementCC3D("NeighborOrder",{},"1")
    
    contact=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Contact"})
    contact.ElementCC3D("NeighborOrder",{},"4")
    
    contact.ElementCC3D("Energy",{"Type1":"Medium","Type2":"Medium"}, 1.) #The Energy class specifies contact energies between compartments in different cells
    contact.ElementCC3D("Energy",{"Type1":"Medium","Type2":"SUBS_A"}, J0)
    contact.ElementCC3D("Energy",{"Type1":"Medium","Type2":"SUBS_NA"}, -J0)
    contact.ElementCC3D("Energy",{"Type1":"Medium","Type2":"CYTO"}, J0)
    contact.ElementCC3D("Energy",{"Type1":"Medium","Type2":"FRONT"}, 2.*J0/3)
    contact.ElementCC3D("Energy",{"Type1":"Medium","Type2":"NUCL"}, 5.*J0)
    #
    contact.ElementCC3D("Energy",{"Type1":"SUBS_A","Type2":"SUBS_A"}, 1.)
    contact.ElementCC3D("Energy",{"Type1":"SUBS_A","Type2":"SUBS_NA"}, 1.)
    contact.ElementCC3D("Energy",{"Type1":"SUBS_A","Type2":"CYTO"}, J0)
    contact.ElementCC3D("Energy",{"Type1":"SUBS_A","Type2":"FRONT"}, J0/3)
    contact.ElementCC3D("Energy",{"Type1":"SUBS_A","Type2":"NUCL"}, 5.*J0)
    #
    contact.ElementCC3D("Energy",{"Type1":"SUBS_NA","Type2":"SUBS_NA"}, 1.)
    contact.ElementCC3D("Energy",{"Type1":"SUBS_NA","Type2":"CYTO"}, J0)
    contact.ElementCC3D("Energy",{"Type1":"SUBS_NA","Type2":"FRONT"}, 2.*J0/3)
    contact.ElementCC3D("Energy",{"Type1":"SUBS_NA","Type2":"NUCL"}, 5.*J0)
    # 
    contact.ElementCC3D("Energy",{"Type1":"CYTO","Type2":"CYTO"}, 2.*J0)
    contact.ElementCC3D("Energy",{"Type1":"CYTO","Type2":"FRONT"}, 2.*J0)
    contact.ElementCC3D("Energy",{"Type1":"CYTO","Type2":"NUCL"}, 5.*J0)
    # 
    contact.ElementCC3D("Energy",{"Type1":"FRONT","Type2":"FRONT"}, 2.*J0)
    contact.ElementCC3D("Energy",{"Type1":"FRONT","Type2":"NUCL"}, 5.*J0)
    # 
    contact.ElementCC3D("Energy",{"Type1":"NUCL","Type2":"NUCL"}, 5.*J0)
    #
    # The "ContactInternal" plugin loads the calculation of the compartment-compartment contact energies in equation 7, 
    # when these compartments belong to the SAME cell
    #
    intContact=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"ContactInternal"})
    intContact.ElementCC3D("NeighborOrder",{},"4")    
    # 
    intContact.ElementCC3D("Energy",{"Type1":"CYTO","Type2":"CYTO"}, 0.)
    intContact.ElementCC3D("Energy",{"Type1":"CYTO","Type2":"FRONT"}, J0/2) 
    intContact.ElementCC3D("Energy",{"Type1":"CYTO","Type2":"NUCL"}, J0)
    # 
    intContact.ElementCC3D("Energy",{"Type1":"FRONT","Type2":"FRONT"}, 0.0)
    intContact.ElementCC3D("Energy",{"Type1":"FRONT","Type2":"NUCL"}, 2.*J0)
    # 
    intContact.ElementCC3D("Energy",{"Type1":"NUCL","Type2":"NUCL"}, 0.)
    
    PluginElmnt_4=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Chemotaxis"})
    ChemicalFieldElmnt=PluginElmnt_4.ElementCC3D("ChemicalField",{"Name":"FActin", "Source":"DiffusionSolverFE"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"ChemotactTowards":"Medium","Type":"FRONT","Lambda":lambCHEM})
    
    PluginElmnt_5=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Secretion"})
    FieldElmnt=PluginElmnt_5.ElementCC3D("Field",{"Name":"FActin"})
    FieldElmnt.ElementCC3D("SecretionOnContact",{"SecreteOnContactWith":"SUBS_NA","Type":"FRONT"},0.9)
    
    SteppableElmnt=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"DiffusionSolverFE"})
    
    DiffusionFieldElmnt=SteppableElmnt.ElementCC3D("DiffusionField",{"Name":"FActin"})
    DiffusionDataElmnt=DiffusionFieldElmnt.ElementCC3D("DiffusionData")
    DiffusionDataElmnt.ElementCC3D("FieldName",{},"FActin")
    DiffusionDataElmnt.ElementCC3D("GlobalDiffusionConstant",{},1e-04)
    DiffusionDataElmnt.ElementCC3D("GlobalDecayConstant",{},0.9)
    
    SecretionDataElmnt=DiffusionFieldElmnt.ElementCC3D("SecretionData")
    SecretionDataElmnt.ElementCC3D("SecretionOnContact",{"SecreteOnContactWith":"SUBS_A","Type":"FRONT"},0.9) 

    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)    


configure_simulation()

from CellMig3D_P3_Steppables import Cell # Reads the "Cell" steppable from the file CellMig3D_P3_Steppables.py. This steppable creates the cell and updates its compartments
cellInstance=Cell(frequency=1,phiF=phiF,phiN=phiN,cellRad=cellRad,cellVol=cellVol,x1=x1,y1=y1) # Instantiates the "Cell" steppable, directs the engine to call it once per mcs (_frequency=1), 
# and passes to it the parameter values given above for phiF, phiN, cellRad, cellVol, x1 and y1
CompuCellSetup.register_steppable(steppable=cellInstance) # Registers the "Cell" steppable so it is called by the simulation engine
#
from CellMig3D_P3_Steppables import Calc # Reads the "Calc" steppable from the file CellMig3D_P3_Steppables.py
calcInstance=Calc(frequency=1,phiF=phiF,phiN=phiN,lambCHEM=lambCHEM,cellRad=cellRad,g1Flag=g1Flag,g2Flag=g2Flag,x1=x1,y1=y1,dT=deltaT)# Instantiates the "Calc" steppable, directs the engine to call it once per mcs (_frequency=1), 
# and passes to it the parameter values given above for phiF, phiN, lambdaCHEM, cellRad, g1Flag, g2Flag, x1, y1 and deltaT
CompuCellSetup.register_steppable(steppable=calcInstance) # Registers the "Calc" steppable so it is called by the simulation engine

CompuCellSetup.run()
