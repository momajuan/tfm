#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:06:14 2021

@author: jj
"""

import os
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
# Import required package
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import random
from collections import Counter
def Extraccionmapa(nombre_mapa, nombre_ruta,file_out_rutas):
    dom=ET.parse(nombre_mapa)
    root = dom.getroot()
    edgesmap=[]
    v=root.findall('./edge')
    for elm in v:
        edgesmap.append(str(elm.attrib["id"]))
    dom_rutas=ET.parse(nombre_ruta)
    root_rutas = dom_rutas.getroot()
    v=root_rutas.findall('./route')
    rutas=[]
    j=0
    v=root_rutas.findall('./route')
    for elm in v:
        j+=1
        ruta=elm.attrib["edges"]
        ruta=ruta.split()
        ruta_nueva=[]
        for i in range(len(ruta)):
            if ruta[i] in edgesmap:
                ruta_nueva.append(ruta[i])
            elif len(ruta_nueva)!=0:
                i=i-1
                break
        if len(ruta_nueva)==0 or len(ruta_nueva)==1:
            root_rutas.remove(elm)
        else:
            elm.set("edges", " ".join(ruta_nueva))
            elm.set("n_aris_ant", str(i- len(ruta_nueva)+1))
            elm.set("id_viejo", elm.attrib["id"])
            elm.set("id", "_to_".join([ruta_nueva[0], ruta_nueva[-1]]))
    rutas_nuevas=[]
    for i in dom_rutas.findall('./route'):
        rutas_nuevas.append(i.attrib["id"])
        
    num_repetidas=Counter(rutas_nuevas)
    v=dom_rutas.findall('./route')        
    dicc_compar={}
    dic_rutas_repetidas={}
    for i in v:
        if num_repetidas[i.attrib["id"]]>1:
            if (i.attrib["id"] in dic_rutas_repetidas.keys())== False:
                dic_rutas_repetidas.update({i.attrib["id"]:0})
            else:
                dic_rutas_repetidas[i.attrib["id"]]+=1
            i.set("id", i.attrib["id"]+"#"+str(dic_rutas_repetidas[i.attrib["id"]]))
        dicc_compar.update({i.attrib["id_viejo"]:{"id":i.attrib["id"],"n_aris_antes":i.attrib["n_aris_ant"],"rutass":i.attrib["edges"]}})
    
    dom_rutas.write(file_out_rutas)
    return dicc_compar

def extraccionruta(dicc,prob,nombre_add,file_out_add):
    dom=ET.parse(nombre_add)
    root = dom.getroot()
    v=root.findall('./vehicle')
    for i in v:
        if prob > random.random() and i.attrib["route"] in dicc:
            # pass
            # break
            o=ET.SubElement(i, "route")
            o.set("edges",dicc[i.attrib["route"]]["rutass"])
            i.set("depart",str(float(i.attrib["depart"])+20* float(dicc[i.attrib["route"]]["n_aris_antes"])))
            i.attrib.pop("route")
        else:
            root.remove(i)
    v=root.findall('./vehicle')
    
    b=root.findall("./routeDistribution")
    
    for elm_routdist in b:
        k=elm_routdist.findall("./route")
        for l in k:
            if l.attrib["refId"] in dicc:
                l.set("refId", dicc[l.attrib["refId"]]["id"])
            else:
                elm_routdist.remove(l)
        
      
    v=root.findall('./vehicle')
    
    b=root.findall("./routeDistribution")
    
    for elm_routdist in b:
        k=elm_routdist.findall("./route")
        if True:
            root.remove(elm_routdist)
    dom.write(file_out_add) 
    
