# !/usr/bin/env python
# -*- coding: utf-8 -*-

'a decoder module'

__author__ = 'Hall Wei'


from PIL import Image
import numpy as np
import math

class Decoder(object):
    def __init__(self, compress_file, huffman_root, quant_table,seg_number, N1, N2, height, width):
        self.height = height
        self.width = width
        self.seg_number = seg_number
        self.N1 = N1
        self.N2 = N2
        self.huffman_root = huffman_root
        self.compress_file = compress_file
        
        self.quant_table = quant_table

        self.huffman_decoder_res = []
        self.quant_decoder_res = []
        self.IDCT_res = []
        self.restruction_res = []
        
    
    def huffman_decoder(self, input_data, huffman_root, seg_number, N1, N2):
        current_node = huffman_root
        h_decoder_res = []
        l = len(input_data)
        index = 0
        while index < l:
            if current_node.left == -1:
                h_decoder_res.append(current_node.value)
                current_node = huffman_root
            elif input_data[index] == '1':
                current_node = current_node.left
                index += 1
            else:
                current_node = current_node.right
                index += 1
        h_decoder_res.append(current_node.value)
        output_block = np.array(h_decoder_res).reshape((seg_number, N1, N2))
        return output_block


    def decoder_quant_res(self, input_data, seg_number,quant_table):
        res = []
        for i in range(seg_number):
            temp = input_data[i] * quant_table
            res.append(temp)
        return res

    def IDCT(self, input_data, seg_number, N1, N2):
        PI = math.pi
        res = []
        # get matrix C
        c = np.zeros((N1, N2), dtype = np.float)
        for i in range(N1):
            for j in range(N2):
                if i == 0:
                    c_u = math.sqrt(1/N1)
                else:
                    c_u = math.sqrt(2/N1)
                if j == 0:
                    c_v = math.sqrt(1/N2)
                else:
                    c_v = math.sqrt(2/N2)
                c[i][j] = c_u * c_v
        
        cos_coff = []
        for i in range(N1):
            for j in range(N2):
                cos_a = np.array(list(math.cos((i + 0.5) * PI / N1 * u) for u in range(N1)))
                cos_b = np.array(list(math.cos((j + 0.5) * PI / N2 * v) for v in range(N2)))
                cos_a = cos_a.reshape(N1, 1)
                cos_coff.append(cos_a * cos_b)

        for index in range(seg_number):
            temp = np.zeros((N1, N2), dtype = float)
            for i in range(N1):
                for j in range(N2):
                    temp[i][j] = np.sum(input_data[index] * cos_coff[i * N1 + j] * c)
                    temp[i][j] = temp[i][j].astype(np.uint8)
            res.append(temp)
        return res

    def combine(self, input_data, seg_number, N1, N2, height, width):
        res = np.zeros((height, width))
        row_begin = 0
        index = 0
        while row_begin < height:
            col_begin = 0
            while col_begin < width:
                res[row_begin:(row_begin + N1), col_begin:(col_begin + N2)] = input_data[index]
                index += 1
                col_begin += N2
            row_begin += N1
        return res

    def start(self):
        self.huffman_decoder_res = self.huffman_decoder(self.compress_file, self.huffman_root, self.seg_number, self.N1, self.N2)
        self.quant_decoder_res = self.decoder_quant_res(self.huffman_decoder_res,self.seg_number,self.quant_table)
        self.IDCT_res = self.IDCT(self.quant_decoder_res, self.seg_number, self.N1, self.N2)
        self.restruction_res = self.combine(self.IDCT_res, self.seg_number, self.N1, self.N2, self.height, self.width)
        return 1