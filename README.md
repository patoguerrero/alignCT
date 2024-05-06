# alignCT
python codes of the paper 'Automatic and computationally efficient alignment in fan- and cone-beam tomography',
IEEE Transactions on Computational Imaging 2024 and arXiv:2310.09567.  
Patricio Guerrero

if you only want to obtain the alignment parameters, it needs:  
numpy, scipy, skimage  

if you want to simulate and reconstruct (and visualise) CT data, you also need:  
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





