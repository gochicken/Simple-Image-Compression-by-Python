# !/usr/bin/env python
# -*- coding: utf-8 -*-

'a encoder module'

__author__ = 'Hall Wei'


from PIL import Image
import numpy as np
import math

class Encoder(object):
    def __init__(self, image_address, compress_address, N1 = 8, N2 = 8):
        self.image_address = image_address
        self.compress_address = compress_address
        self.N1 = N1
        self.N2 = N2

        self.raw_image = []
        self.height = 0
        self.width = 0

        self.seg_file = []
        self.seg_number = 0

        self.dct_res = []
        self.quant_table = []
        
        self.quant_res = []


    #######################################
    #This is a image reading funx
    #-------------------------------------#
    #INPUT :     address of the image
    #OUTPUT :    raw_image, height, width
    #raw_image : image data saved in list
    #height:     height of the image
    #width:      width of the image
    #######################################

    def image_reading(self, image_address):
        im = Image.open(image_address).convert('L')
        res = np.array(im)
        height, width = res.shape
        height = height - height % 8
        width = width - width % 8
        raw_image = res[0:height][0:width]
        im.close()
        return raw_image, height, width

    #######################################
    #This funx is in order to seg image in to pieces
    #Each pieces' size is N1 * N2
    #-------------------------------------#
    #INPUT:        image, height, width, N1, N2
    #image:        the input data
    #height:       height of the image
    #width:        width of the image
    #N1:           row number of each piece
    #N2:           col number of each piece
    #-------------------------------------#
    #OUTPUT:       res, l
    #res:          the data after processing
    #l:            the piece's number
    ########################################

    def seg(self, image, height, width, N1, N2):
        res = []
        row_begin = 0
        while row_begin < height:
            col_begin = 0
            while col_begin < width:
                #print(col_begin)
                res.append(image[row_begin:(row_begin + N1),col_begin:(col_begin + N2)])
                col_begin += N2
            row_begin += N1
        return res, len(res)
    #######################################
    #This funx implements DCT
    #-------------------------------------#
    #INPUT: input_data, length,N1, N2
    #input_data: the length of input_data list is length, each element is a N1*N2 matrics
    #N1: row number of each element
    #N2: col number of each element
    #OUTPUT: res
    #res: new data after DCT
    #######################################
    def DCT(self, input_data, seg_number, N1, N2):
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
        for u in range(N1):
            for v in range(N2):
                cos_a = np.array(list(math.cos((i + 0.5) * PI / N1 * u) for i in range(N1)))
                cos_b = np.array(list(math.cos((i + 0.5) * PI / N2 * v) for i in range(N2)))
                cos_a = cos_a.reshape(N1, 1)
                cos_coff.append(cos_a * cos_b)

        for index in range(seg_number):
            temp = np.zeros((N1, N2), dtype = float)
            for u in range(N1):
                for v in range(N2):
                    temp[u][v] = np.sum(input_data[index] * cos_coff[u * N1 + v]) * c[u][v]
                    #temp[u][v] = temp[u][v].astype(np.int)
            res.append(temp)
        return res
    
    def quantification(self, input_data, len_input_data, quant_table):
        res = []
        for i in range(len_input_data):
            if isinstance(input_data[i], list):
                dividend = np.array(input_data[i])
            else:
                dividend = input_data[i]
            temp = (dividend/quant_table).astype(np.int)
            res.append(temp)
        return res

    def quant_table_reading(self, address):
        f = open(address, 'r')
        table = f.read()
        f.close()
        res = []
        temp = []
        current_number = 0
        for char in table:
            if char == '\n':
                temp.append(current_number)
                res.append(temp)
                current_number = 0
                temp = []
            elif char == ' ':
                temp.append(current_number)
                current_number = 0
            else:
                current_number = current_number * 10 + int(char)
        temp.append(current_number)
        res.append(temp)
        return np.array(res)

    def huffman_build_tree(self, input_data, length, N1, N2):
        d = {}
        for i in range(length):
            for n_1 in range(N1):
                for n_2 in range(N2):
                    if d.get(input_data[i][n_1][n_2], -1) == -1:
                        d[input_data[i][n_1][n_2]] = 1
                    else:
                        d[input_data[i][n_1][n_2]] += 1
        class node(object):
            def __init__(self,key,value,left, right, flag):
                self.key = key
                self.value = value
                self.left = left
                self.right = right
                self.flag = flag

        table = []
        for i in d.keys():
            temp = node(d[i], i, -1, -1, False)
            table.append(temp)
        
        table = sorted(table, key = lambda x : x.key)
        while len(table) != 1:
            temp = node(table[0].key + table[1].key, -1, table[0], table[1], False)
            table.pop(0)
            table.pop(0)
            index = 0
            for i in table:
                if i.key < temp.key:
                    index += 1
                else:
                    break
            
            table.insert(index, temp)
        return table[0]
    
    def generate_huffman_table(self, root):
        st = [root]
        current_key = ''
        huffman_table = {}
        while len(st) > 0:
            #print(st[-1].key)
            if st[-1].left == -1:
                huffman_table[st[-1].value] = current_key
                st[-1].flag = True
                st.pop()
                #print(len(st))
            else:
                if st[-1].left.flag == False:
                    #print("left key is %d" % st[-1].left.key)
                    st[-1].key = current_key
                    #print("now key is")
                    #print(current_key)
                    current_key += '1'
                    st.append(st[-1].left)
                    
                elif st[-1].right.flag == False:
                    #print("right, key is")
                    #print(st[-1].key)
                    current_key = st[-1].key
                    current_key += '0'
                    st.append(st[-1].right)
                    
                else:
                    st[-1].flag = True
                    st.pop()
        
        return huffman_table

    def generate_compressed_file(self, huffman_table, quant_res, seg_number, N1, N2):
        trans_codes = ''
        for i in range(seg_number):
            for n_1 in range(N1):
                for n_2 in range(N2):
                    trans_codes += huffman_table[quant_res[i][n_1][n_2]]
        #f = open('compress_lena', 'wb')
        #f.write(trans_codes)
        #f.close()
        return trans_codes, len(trans_codes)





    def start(self):
        self.raw_image, self.height, self.width = self.image_reading(self.address)
        print("image reading finished")
        print("the height of image is %d, the width of image is %d" % (self.height, self.width))

        self.seg_file, self.seg_number = self.seg(self.raw_image, self.height, self.width, self.N1, self.N2)       
        print("image seg finished")
        print("the number of pieces is %d" %self.seg_number)

        self.dct_res = self.DCT(self.seg_file, self.seg_number, self.N1, self.N2)
        print("DCT finished")

        quant_table_address = 'quant_table.txt'
        self.quant_table = self.quant_table_reading(quant_table_address)
        print("quant_table loading finished")

        self.quant_res = self.quantification(self.dct_res, self.seg_number, self.quant_table)
        print("quantification finished")

        huffman_tree_root = self.huffman_build_tree(self.quant_res, self.seg_number, self.N1, self.N2)
        print("huffman_tree building finished")

        huffman_table = self.generate_huffman_table(huffman_tree_root)
        print("huffman table generating finished")

        trans_codes, bit_number = self.generate_compressed_file(huffman_table, self.quant_res,self.seg_number, self.N1, self.N2)
        print("transcode_generate finished, the final bit number is %d" % bit_number)

        return trans_codes, huffman_tree_root
    





    