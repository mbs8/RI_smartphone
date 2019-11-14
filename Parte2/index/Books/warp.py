import requests
import sys
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm
import time
import numpy as np
import pandas as pd
import nltk   
import unicodedata
from html.parser import HTMLParser
import re
from tqdm import tqdm
import glob
from utils import *
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import requests
from lxml import etree
import re

most_frequents = {
    'price':['preço', 'preco'],
    'model':['modelo'],
    'ram':['memoria ram', 'ram', 'memória RAM'],
    'hd':['armazenamento interno', 'memória interna', 'memoria interna', 'interna'],
    'screen':['tamanho da tela', 'tela', 'tamanho do display', 'display', 'tamanho', 'tamanho (tela principal)'],
    'color':['cor']
}


latin_dict = \
            {'\\u00e1':'a', '\\u00e0':'a', '\\u00e2':'a', '\\u00e3':'a', '\\u00e4':'a', '\\u00c1':'A', '\\u00c0':'A', 
             '\\u00c2':'A', '\\u00c3':'A', '\\u00c4':'A', '\\u00e9':'e', '\\u00e8':'e', '\\u00ea':'e', '\\u00ea':'e', 
             '\\u00c9':'E', '\\u00c8':'E', '\\u00ca':'E', '\\u00cb':'E', '\\u00ed':'i', '\\u00ec':'i', '\\u00ee':'i', 
             '\\u00ef':'i', '\\u00cd':'I', '\\u00cc':'I', '\\u00ce':'I', '\\u00cf':'I', '\\u00f3':'o', '\\u00f2':'o', 
             '\\u00f4':'o', '\\u00f5':'o', '\\u00f6':'o', '\\u00d3':'O', '\\u00d2':'O', '\\u00d4':'O', '\\u00d5':'O', 
             '\\u00d6':'O', '\\u00fa':'u', '\\u00f9':'u', '\\u00fb':'u', '\\u00fc':'u', '\\u00da':'U', '\\u00d9':'U', 
             '\\u00db':'U', '\\u00e7':'c', '\\u00c7':'C', '\\u00f1':'n', '\\u00d1':'N', '\\u0026':'E', '\\u0027':'\'',
             'á':'a', 'à':'a', 'â':'a', 'ã':'a', 'ä':'a', 'Á':'A', 'À':'A', 'Â':'A', 'Ã':'A', 'Ä':'A', 'é':'e', 'è':'e', 
             'ê':'e', 'ê':'e', 'É':'E', 'È':'E', 'Ê':'E', 'Ë':'E', 'í':'i', 'ì':'i', 'î':'i', 'ï':'i', 'Í':'I', 'Ì':'I', 
             'Î':'I', 'Ï':'I', 'ó':'o', 'ò':'o', 'ô':'o', 'õ':'o', 'ö':'o', 'Ó':'O', 'Ò':'O', 'Ô':'O', 'Õ':'O', 'Ö':'O', 
             'ú':'u', 'ù':'u', 'û':'u', 'ü':'u', 'Ú':'U', 'Ù':'U', 'Û':'U', 'ç':'c', 'Ç':'C', 'ñ':'n', 'Ñ':'N', '&':'E'}


def levenshtein(word1, word2):
    """
    https://medium.com/@yash_agarwal2/soundex-and-levenshtein-distance-in-python-8b4b56542e9e
    https://en.wikipedia.org/wiki/Levenshtein_distance
    :param word1:
    :param word2:
    :return:
    """
    word2 = word2.lower()
    word1 = word1.lower()
    matrix = [[0 for x in range(len(word2) + 1)] for x in range(len(word1) + 1)]

    for x in range(len(word1) + 1):
        matrix[x][0] = x
    for y in range(len(word2) + 1):
        matrix[0][y] = y

    for x in range(1, len(word1) + 1):
        for y in range(1, len(word2) + 1):
            if word1[x - 1] == word2[y - 1]:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1],
                    matrix[x][y - 1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,
                    matrix[x - 1][y - 1] + 1,
                    matrix[x][y - 1] + 1
                )

    return matrix[len(word1)][len(word2)]

def nomr_l(word1, word2):
    return levenshtein(word1, word2) / max(len(word1), len(word2))

def html_to_np(table):
    result = []
    all_rows = table.find_all('tr')
    for row in all_rows:
        result.append([])
        all_cols = row.find_all('td')
        for col in all_cols:
            col_text = [s.strip(' ') for s in col.find_all(text=True)]
            col_text  = ''.join(col_text) # it should not change anything
            result[-1].append(col_text)
    return np.array(result)


def np_to_dict(table, fields_dict):
    new_dict = fields_dict.copy()
    for cols in table:
        if len(cols) != 2:
            continue
        for key in fields_dict.keys():
            if nomr_l(key, cols[0]) <= 0.25:
                new_dict[key] = cols[1].replace("\n",'')
            #print(nomr_l(key, cols[0]), key, cols[0])
    return new_dict


def get_fields_amazon(html_page):
    fields_amazon = {
            'Preço': '',
            'Sistema operacional': '',
            'RAM': '',
            'Capacidade de armazenamento digital': '',
            'Modelo': '',
            'Cor': '',  
            'Tamanho da tela':'',
            'Avaliações de clientes': ''
           }
    
    parser = etree.HTMLParser()
    XPATH_TABLE1 = """//*[@id="prodDetails"]/div[2]/div[1]/div/div[2]/div/div/table/tbody/tr"""
    XPATH_TABLE2 = """//*[@id="prodDetails"]/div[2]/div[2]/div/div[2]/div/div/table/tbody/tr"""
    XPATH_AVAL = """//*[@id="prodDetails"]/div[2]/div[2]/div/div[2]/div/div/table/tbody/tr[10]/td[2]/span/span[3]/a/text()"""
    PRICE = """//*[@id="price"]/table/tr/td/span[1]/text()"""
    tree = etree.parse(html_page, parser)
    RAW_TABLE1 = tree.xpath(XPATH_TABLE1)
    RAW_TABLE2 = tree.xpath(XPATH_TABLE2)
    RAW_AVAL = tree.xpath(XPATH_AVAL)
    RAW_PRICE = tree.xpath(PRICE)
    
    try:
        fields_amazon['Preço'] = RAW_PRICE[0]
    except:
        pass
    
    try:
        fields_amazon = parse_table_amazon(fields_amazon, RAW_TABLE1)
        fields_amazon = parse_table_amazon(fields_amazon, RAW_TABLE2)
    except:
        pass
    
    try:
        fields_amazon['Avaliações de clientes'] = RAW_AVAL[0].strip()
    except:
        pass
                
    return fields_amazon

def parse_table_amazon(fields, table):
    try:
        for row in table:
            for key in fields:
                att = row[0].text.replace(" ", "")
                loss = nomr_l(key, att)
                if(loss < 0.3):
                    value = row[1].text
                    fields[key] = value
    except:
        pass
    
    return fields

def get_fields_taqi(html_page):
    fields_taqi = {
            'Preço': '',
            'Sistema operacional': '',
            'Memória RAM': '',
            'Memória Interna': '',
            'Modelo': '',
            'Cor': '',  
            'Tamanho':'',
            'Avaliações de clientes': ''
           }
    
    parser = etree.HTMLParser()
    XPATH_DD = """//*[@id="descricaoproduto"]/div[2]/span/p/span/text()"""
    PRICE = """//*[@id="lowPrice"]"""
    tree = etree.parse(html_page, parser)
    RAW_DD = tree.xpath(XPATH_DD)
    RAW_PRICE = tree.xpath(PRICE)
    
    try:
        fields_taqi['Preço'] = RAW_PRICE[0].text
    except:
        pass
    
    
    for line in RAW_DD:
        try:
            att, value = line.split(":")
            if(value[-1] == "."):
                value = value[:-1]
            for key in fields_taqi:
                loss = nomr_l(key, att)
                if(loss < 0.2):
                    fields_taqi[key] = value
        except:
            pass
                
    return fields_taqi

def get_fields_havan(html_page):
    fields_havan = {
            'Preço': '',
            'Sistema operacional': '',
            'Memória RAM': '',
            'Armazenamento Total': '',
            'Modelo': '',
            'Cor': '',  
            'Tamanho da tela':'',
            'Avaliações de clientes': ''
           }
    
    parser = etree.HTMLParser()
    XPATH_DD = """//*[@id="description"]/div/div/text()"""
    PRICE = """//*[@id="maincontent"]/div[2]/div/div[1]/div[2]/div[3]/span/span/span/text()"""
    
    tree = etree.parse(html_page, parser)
    RAW_DD = tree.xpath(XPATH_DD)
    RAW_PRICE = tree.xpath(PRICE)
    
    if (len(RAW_PRICE) >= 1):
        fields_havan['Preço'] = RAW_PRICE[0]
        
    for line in RAW_DD:
        try:
            att, value = line.split(":")
            if(value[-1] == "."):
                value = value[:-1]
            for key in fields_havan:
                loss = nomr_l(key, att)
                if(loss < 0.2):
                    fields_havan[key] = value
        except:
            pass

    return fields_havan

def get_fields_luiza(html_page):

    fields_luiza = {
            'Preço': '',
            'Sistema operacional': '',
            'Memória RAM': '',
            'Memória': '',
            'Modelo': '',
            'Cor': '',  
            'Tamanho da tela':'',
            'Avaliações de clientes': '',
            'Marca' : ''
           }
    
    parser = etree.HTMLParser()
    XPATH_TABLE1 = """//*[@id="anchor-description"]/div/table[1]/tr/td/table/tr"""
    XPATH_TABLE2 = """//*[@id="anchor-description"]/div/table[2]/tr/td/table/tr"""
    PRICE = """/html/body/div[3]/div[5]/div[1]/div[4]/div[2]/div[4]/div/div/div/span"""
    tree = etree.parse(html_page, parser)
    RAW_TABLE1 = tree.xpath(XPATH_TABLE1)
    RAW_TABLE2 = tree.xpath(XPATH_TABLE2)
    RAW_PRICE = tree.xpath(PRICE)
    
    if(len(RAW_PRICE) >= 2):
        fields_luiza["Preço"] = "{} {}".format(RAW_PRICE[0].text, RAW_PRICE[1].text) 
    
    
    fields_luiza = parse_table_luiza(fields_luiza, RAW_TABLE1)
    fields_luiza = parse_table_luiza(fields_luiza, RAW_TABLE2)
                
    return fields_luiza

def parse_table_luiza(fields, table):
    try:
        for row in table:
            for key in fields:
                att = row[0].text.replace(" ", "")
                loss = nomr_l(key, att)
                if(loss < 0.3):
                    value = row[1].text
                    fields[key] = value
    except:
        pass
    return fields

def get_fields_ibyte(html_page):
    fields_ibyte = {
            'Preço': '',
            'Sistema operacional': '',
            'Memória RAM(GB)': '',
            'Memória Total Interna (GB)': '',
            'Modelo': '',
            'Cor': '',  
            'Tamanho (Tela Principal)':'',
            'Avaliações de clientes': '',
            'Marca' : ''
           }
    
    parser = etree.HTMLParser()
    XPATH_DD = """//*[@id="descricao"]/table/tbody/tr"""
    PRICE = """//*[@id="product_addtocart_form"]/div[3]/div[1]/div/div/p[2]"""
    tree = etree.parse(html_page, parser)
    RAW_DD = tree.xpath(XPATH_DD)
    RAW_PRICE = tree.xpath(PRICE)
    
    try:
        fields_ibyte['Preço'] = RAW_PRICE[0][1].text.replace("\n", "").replace(" ", "")
    except:
        pass
    
    try:
        for row in RAW_DD:
            if(len(row) == 3):
                att = row[1][0].text
                value  = row[2].text
                for key in fields_ibyte.keys():
                    loss = nomr_l(key, att)
                    if loss < 0.2:
                        fields_ibyte[key] = value
    except:
        pass

    return fields_ibyte


def get_fields_avenida(html_page):
    fields_avenida = {'Preço': '',
        'Sistema operacional': '',
        'Memória RAM': '',
        'Memória Interna': '',
        'Modelo': '',
        'Cor': '',  
        'Tamanho do Display':'',
        'Avaliações de clientes': '',
        'Marca' : ''
       }

    parser = etree.HTMLParser()
    XPATH_DD = """//*[@id="caracteristicas"]/table"""
    PRICE = """//*[@id="product-content"]/div[2]/div[1]/div[2]/div/div[3]/div/p/em/strong"""
    tree = etree.parse(html_page, parser)
    RAW_DD = tree.xpath(XPATH_DD)
    RAW_DD = RAW_DD[0]
    RAW_PRICE = tree.xpath(PRICE)
    RAW_PRICE = RAW_PRICE
    
    try:
        fields_avenida['Preço'] = RAW_PRICE[1].text
    except:
        pass
    try:
        for key in fields_avenida.keys():
            for element in RAW_DD:
                loss = nomr_l(element[0].text, key)
                if loss < 0.4:
                    fields_avenida[key] = element[1].text
    except:
        pass

    return fields_avenida

def get_fields_cissamagazine(html_page):
    fields_cissamagazine = {
            'Preço': '',
            'Sistema operacional': '',
            'Memória RAM': '',
            'Memória Interna': '',
            'Modelo': '',
            'Cor': '',  
            'Tamanho da tela':'',
            'Avaliações de clientes': '',
            'Marca' : ''
           }

    parser = etree.HTMLParser()
    XPATH_DD = r"""//*[@id="caracteristicas"]/div[2]/div/dl/span/dd/text()"""
    XPATH_DT = r"""//*[@id="caracteristicas"]/div[2]/div/dl/span/dt/text()"""
    PRICE = """//*[@id="content-price-product"]/div[1]/div[2]/div/span[1]/span[1]/text()"""
    tree = etree.parse(html_page, parser)
    RAW_DD = tree.xpath(XPATH_DD)
    RAW_DT = tree.xpath(XPATH_DT)
    RAW_PRICE = tree.xpath(PRICE)
    
    for p in RAW_PRICE:
        if len(p):
            price = (re.findall("R\$ \d*.\d{1,5}\,\d{1,3}", str(p)))
            if len(price):
                fields_cissamagazine["Preço"] = price[0]


    try:
        for p in zip(RAW_DT, RAW_DD):
            for key in fields_cissamagazine.keys():
                loss = nomr_l(p[0], key)
                if loss < 0.2:
                    fields_cissamagazine[key] = p[1]
    except:
        pass

    return fields_cissamagazine





#####################################################################################################
most_frequents = {
    'price':['preço', 'preco'],
    'model':['modelo'],
    'ram':['memoria ram', 'ram', 'memória RAM'],
    'hd':['armazenamento interno', 'memória interna', 'memoria interna', 'interna', 'Interna total compartilhada',
          'Memória ROM (Flash) (Armaz. Interno)', 'Memória ROM'],
    'screen':['tamanho da tela', 'tela', 'tamanho do display', 'display', 'tamanho', 'tamanho (tela principal)'],
    'color':['cor']
}
geral_fields = {
            'price':'',
            'model':'',
            'ram':'',
            'hd':'',
            'screen':'',
    }

def get_fields_kabum(html_page):

    def get_price(fields, X_path, tree, verbose=False):

        path = tree.xpath(X_path)

        if len(path) != 0:
            fields['price'] =  path[0].text



    def get_att(fields, X_path, tree):
        path = tree.xpath(X_path)

        for p in path:
            if(p.text != None):
                info = p.text.split(":")
                if len(info) == 2:
                    attrib, value = info
                    attrib = attrib.replace("- ", "") 

                    for key in most_frequents.keys():
                        for key_name in most_frequents[key]:
                            loss = nomr_l(key_name, attrib)
                            if loss < 0.4:
                                fields[key] = value.replace("\"", "").replace("\'", "")
                                break
        #return fields
    fields = geral_fields.copy()

    PRICE        = """//*[@id="pag-detalhes"]/div/div[2]/div[2]/div[2]/div[1]/span/span/span/strong"""
    PRICE_PROMO  = """//*[@id="pag-detalhes"]/div/div[2]/div[2]/div[1]/div[3]/div[4]/span[1]/span"""
    ATT          = """//*[@id="pag-detalhes"]/div/div[6]/div[2]/p"""
    ATT_PROMO    = """//*[@id="pag-detalhes"]/div/div[5]/div[2]/p"""

    parser = etree.HTMLParser()
    tree = etree.parse(html_page, parser)

    get_price(fields, PRICE, tree)
    get_price(fields, PRICE_PROMO, tree)
    get_att(fields, ATT, tree)
    get_att(fields, ATT_PROMO, tree)

    
    return fields


def get_fields_ricardo(html_page):
    
    def get_price(fields, X_path, tree, verbose=False):

        path = tree.xpath(X_path)
        if len(path) != 0:
            fields['price'] =  path[0]
            
    def get_att(fields, X_path, tree):
        
        path = tree.xpath(X_path)
        for idx, p in enumerate(path):
            tr = p.findall("tr")
            for y in range(len(tr)):
                propesed_key, propesed_value = tr[y][0], tr[y][1]
                for key in fields.keys():
                    for key_name in most_frequents[key]:

                        loss = nomr_l(propesed_key.text, key_name)
                        if loss <= 0.35:
                            fields[key] = propesed_value.text
            
    
    fields = geral_fields.copy()


    PRICE = r"""//*[@id="ProdutoDetalhesPrecoComprarAgoraPrecoDePreco"]/text()"""
    ATT   = r"""//*[@id="aba-caracteristicas"]/div[1]/table[1]"""
    
    parser = etree.HTMLParser()
    tree = etree.parse(html_page, parser)
    
    
    
    get_price(fields, PRICE, tree)
        
    get_att(fields, ATT, tree)
    
    return fields


def get_fields_colombo(html_page):
    fields = geral_fields.copy()
    
    def get_price(fields, X_path, tree, verbose=False):
        path = tree.xpath(X_path)
        for idx, p in enumerate(path):
            p = p.strip()
            fd = re.findall("\d*.\d{1,5}\,\d{1,3}", str(p))
            if len(fd):
                fields["Preço"] = "R$ " + fd[0]
            
            
    
    parser = etree.HTMLParser()
    ATT    = r"""//*[@id="produto-caracteristicas"]/div/div[2]/div/div/span/text()"""
    PRICE = """//*[@id="dados-produto-disponivel"]/div[2]/span[2]/span/text()"""
    tree = etree.parse(html_page, parser)
    
    get_price(fields, PRICE, tree)
    
    att_path = tree.xpath(ATT)

    try:
        
        for idx, p in enumerate(att_path):
            p = p.strip()
            print(p)
            if idx % 2 == 0:
                for key in fields.keys():
                    loss = nomr_l(p, key)

                    if loss <= 0.25:
                        fields[key] = att_path[idx+1].strip().split('\n')[0]
    except Exception as e: print(e)
                    
    return fields
