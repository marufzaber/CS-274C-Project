Application of Different Neural Networks Techniques to Genre
=====================================================================

Dataset:
https://github.com/mdeff/fma

Instructions:

1. install a virtualenv (python 3.7)
2. Replace the line `pandas` in fma requirements.txt with: `pandas~=0.22.0`
3. pip install -r requirements.txt
4. git clone the fma repo into projects/fma
5. copy modified_fma_utils.py into projects/fma/utils.py (yes it's hacky!)
6. download the fma_small dataset and fma_metadata
7. put AUDIO_DIR and AUDIO_META_DIR in your environment variables
8. run the project.train.train module (see the cmd line args)

References:
1. https://arxiv.org/pdf/1612.01840.pdf  
2. http://static.echonest.com/enspex/
3. https://scholar.google.com/scholar?as_ylo=2019&q=FMA:+A+Dataset+For+Music+Analysis.+arXiv+2017&hl=en&as_sdt=0,5
4. https://arxiv.org/pdf/1906.11783.pdf
5. http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1354738&dswid=157

