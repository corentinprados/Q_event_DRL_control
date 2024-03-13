// ----------------------------------------------------------------
// ----------- MY MODIFICATION TO MAKE MINIMAL CHANNEL ------------
// ----------------------------------------------------------------


// ----------- PARAMETERS  ------------

Re = 180; // Reynolds number

h = 1; // # Common practise as Chebysev polynom but be aware that sometimes it will be 2 or something else (like in SIMSOn)

Lx = 2.67*h; // Length of the domain
Ly = h; // Height of the domain 
Lz = 0.8*h; // Width of the domain 

Nx = 16; // Resolution in the x-direction
Ny = 64; // Resolution in the y-direction # WHY 65 in Luca's paper ? because SIMSON asks see with ALYA if it changes 
Nz = 16; // Resolution in the z-direction

// ----------- MESH --------------

// ----------- Creation of (x,z) regular meshed plan --------------

// Characteristic mesh size 
lc = Lx/Nx;
Printf("dx+ = %g", lc*Re); 

// Define a point at the origin with a characteristic size lc
Point(1) = {0, 0, 0, lc};

// Extrude the point to create a line in the x-direction
line[] = Extrude {Lx, 0, 0.} {
  Point{1}; Layers{Round(Nx)}; 
};

// Extrude the line to create a surface in the z-direction
surface[] = Extrude {0, 0, Lz} {
  Line{line[1]}; Layers{Nz}; Recombine;
};


// ----------- Creation of the semi-Chebysev nodes --------------


Printf("Ny = %g", Ny); 

// Generate vertical positions of mesh points in the y-direction
For i In {0:Ny-2} // # Here Ny-2 include not like Python
  y[i] = 1 - Cos(Pi*(i+1)/(2*Ny)); 
EndFor

// Set specific vertical positions
y[Ny-1] = 1.; // # We ensure that wa have Ly at the end and that the point o goto far or not enough because of approximation 

// Print y+ values for each vertical position
For i In {0:Ny-1}
  Printf("y+[%g] = %g", i, Re*y[i]);
  layer[i] = 1;
EndFor

// ----------- Creation of the all volume --------------

// Create a volumetric mesh based on the generated geometry
volume[] = Extrude {0.0, 1.0, 0.0} {
  Surface{surface[1]}; Layers{layer[], y[]}; Recombine;
};



// ----------- Assignation --------------

// Assign physical entities to surfaces and volumes # See directly on the applciation, it is Gmsh that sets the number associated 
Physical Surface("wall") = {5};
Physical Surface("symmetry") = {27};
Physical Volume("fluid") = {1};


// ----------- Setting of boundary conditions --------------

// Set every surface to be periodic, using surf(1) as master
Physical Surface("Periodic") = {14, 18, 22, 26};


// Generate periodicity between surface pairs
Periodic Surface {18} = {26} Translate {Lx, 0, 0};
Periodic Surface {22} = {14} Translate {0, 0, Lz};

// ----------- Gmsh particular options --------------

// Set the output mesh file version
Mesh.MshFileVersion = 2.2;

// Options controlling mesh generation
Mesh.ElementOrder = 3; // Order 3 mesh
Mesh 3;                // Volumetric mesh
