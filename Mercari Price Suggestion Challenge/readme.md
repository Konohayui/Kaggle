Models:
1. Random Forest
2. Ridge http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634#.WoeQ6YPwaUl
3. Ridge + LGBM https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/lightgbm.pdf
4. Keras (CNN, GRU, FastText)

...* CNN is fast but don't do much, max pooling also
...* GRU is slow but produce some good results
...* FastText is the best obtion for a single model
...* Simple FastText (no preprocessing) + GRU

5. FTRL + FM_FTRL https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
6. FastText + GRU + FM_FTRL
