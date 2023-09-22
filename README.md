# alignCT
python codes of the paper 'Automatic alignment in cone-beam tomography via fan-beam symmetry and variable projection'.  
Patricio Guerrero

if you only want to obtain the alignment parameters, it needs:  
numpy, scipy, skimage  

if you want to simulate and reconstruct (and visualise) CT data:  
astra, multiprocessing, matplotlib  
and a CUDA-compatible GPU card (for astra only)

for fan-beam just run   
```
python3 example2D.py
```
and for cone-beam   
```
python3 example3D.py
```






