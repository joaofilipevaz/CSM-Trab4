# This Python file uses the following encoding: utf-8

# Trabalho 4 CSM
# João Filipe Vaz - 40266
# João Ventura - 38950


import cv2
import numpy as np
# import matplotlib.pyplot as plt
from trab3Test import K3, K5
from time import time
from os import path
import matplotlib



# 1

"""
Considerar que cada frame é uma intra-frame (I). Pode usar o codiﬁcador do trabalho anterior ou usar o codiﬁ-
cador já disponível
"""


import cv2
x_img = cv2.imread("bola_0x.tiff")
cv2.imwrite("bola_0x.jpeg",x_img,(cv2.IMWRITE_JPEG_QUALITY,50))


# 2

"""
Considerar que cada frame à excepção da primeira são inter-frames (P), ou seja, todos os macroblocos são do
tipo (P). Neste codiﬁcador deve criar as P-frames, que são a diferença entre a frame a codiﬁcar e a I-frame, sem
compensação de movimento. Visualize a P-frame (ou seja a imagem a transmitir).
"""


# 3.1

"""
 Considerar que cada frame à excepção da primeira são inter-frames (P). Neste codiﬁcador deve implementar a
predição da frame a codiﬁcar com base na I-frame fazendo a compensação de movimento. A frame a transmitir
é a diferença entre a frame a codiﬁcar e a sua predição. Sugere-se a construção de três funções:

3.1. uma função para fazer a medição do erro absoluto médio entre dois blocos (tamanho 16 × 16);
"""


# 3.2

"""
3.2. uma função que faça uma pesquisa (pode escolher a full-search ou outra) do bloco da frame a codiﬁcar
numa janela de pesquisa (-15 a +15) da I-frame;
"""


# 3.3

"""
3.3. uma função que percorra os blocos da frame a codiﬁcar e construa a frame predita;
Visualizar a frame predita, e a frame diferença, bem como os vectores de movimento (use a função pylab.quiver
para o efeito).
"""

