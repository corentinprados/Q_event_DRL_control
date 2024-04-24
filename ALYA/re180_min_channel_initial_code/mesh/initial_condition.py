#/bin/env python
#
# Create initial condition for ChannelFlow case
from __future__ import print_function, division

import sys, random, numpy as np
import pyAlya

## Flow parameters
CASENAME = sys.argv[1]
rho      = 1.
delta    = 1.
ReTau    = 180.
U0       = 1.
Re       = np.exp((1.0/0.88)*np.log(ReTau/0.09)) # Experimental to obtain Re from ReTau
nu       = (rho*2.0*delta*U0)/Re
utau     = (ReTau*nu)/(delta*rho)
Fx       = utau*utau*rho/delta
print('Re   = ',Re)
print('utau = ',utau)
print('nu   = ',nu)
print('Fx   = ',Fx)


## Obtain XYZ coordinates
# assume turbulent parabolic profile
coordfile = pyAlya.io.MPIO_AUXFILE_S_FMT % (CASENAME,'COORD')
header    = pyAlya.io.AlyaMPIO_header.read(coordfile)
# Read the node coordinates in serial
xyz,_ = pyAlya.io.AlyaMPIO_readByChunk_serial(coordfile,header.npoints,0)

## Create veloc from xyz coordinates
# Generate the velocity array
veloc = np.zeros_like(xyz) # Same dimensions as xyz
veloc[:,0] = -1.0*xyz[:,1]*(xyz[:,1] - 2*delta)/delta**2*U0

# Add a perturbation on veloc of 5% vx
r = np.vstack((veloc[:,0],veloc[:,0],veloc[:,0])).T*0.05*np.random.rand(*xyz.shape)
veloc += r # Add perturbation


## Store velocity as initial condition
outname = pyAlya.io.MPIO_XFLFILE_S_FMT % (CASENAME,1,1) 
h       = pyAlya.io.AlyaMPIO_header(
	fieldname   = 'XFIEL',
	dimension   = 'VECTO',
	association = 'NPOIN',
	dtype       = 'REAL',
	size        = '8BYTE',
	npoints     = header.npoints,
	nsub        = header.nsubd,
	sequence    = header.header['Sequence'],
	ndims       = xyz.shape[1],
	itime       = 0,
	time        = 0.,
	tag1        = 1,
	tag2        = 1,
	ignore_err  = True
)
pyAlya.io.AlyaMPIO_writeByChunk_serial(outname,veloc,h,h.npoints,0)

pyAlya.cr_info()
