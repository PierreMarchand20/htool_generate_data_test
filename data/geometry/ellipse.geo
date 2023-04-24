Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");
h=0.001;
Mesh.CharacteristicLengthMin = h;
Mesh.CharacteristicLengthMax = h;
Disk(1) = {0,0,0,1,0.5};
