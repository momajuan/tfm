#
from __future__ import absolute_import
from __future__ import print_function
exec(open("/home/jj/Documents/tfm/Extraccion1.py").read());
import os
import pandas as pd
from xml.etree import ElementTree as ET
# Import required package
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt 
from scipy.stats import truncnorm
import os
import sys
import optparse
import random
import time
import concurrent.futures
import copy
import shutil
from statistics import mean

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable Juan 'SUMO_HOME'")
# %%
from sumolib import checkBinary  # noqa
import traci  # noqa

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from random import randrange, uniform
import itertools
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable Juan 'SUMO_HOME'")
lr=0.001
device="cuda"
def run():
    """execute the TraCI control loop"""
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        step += 1
    traci.close()
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0 # How many exps have been added to memo

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

           
# %%   
class Extr:
    
   def __init__(self,sem,desfa,jm,df,df_tl,root,dedges):
       self.sem=sem #Pasamos la interseccion a la clase
       self.nombre=self.sem.attrib["id"]
       self.vinc=[] # Vecinos incidentes
       for i in jm[:,df.loc[df['id']==self.sem.attrib['id']].index[0]].nonzero()[0]:
           if i<df_tl.shape[0]:
               self.vinc.append(df_tl.iloc[i][0])
              
       self.vout=[] # Vecinos incidentes
       for i in jm[df.loc[df['id']==self.sem.attrib['id']].index[0],:].nonzero()[1]:
           if i<df_tl.shape[0]:
               self.vout.append(df_tl.iloc[i][0])
       ## Obtenemos las fases de los otros semaforos         
       self.L(root)
       self.accionelegida={}
       self.ultimoestado=list(self.Listaytipo().items())[-1][0]
       self.Acciones(desfa)
       
       self.Prog()
       self.estado=self.Listaytipo()
       self.Acciones(desfa)
       self.tiempoespera=0
       self.tecoloracion={}
       self.Carreteras(dedges)
       self.memory=ReplayMemory(10000)
       self.fasesant=self.fases
    
       #     self.estado.append(selfsita)
       # in_features=


  
## Extracciones de fases que cambiar, ya que hay algunas que son necesarias para mantener la seguridad  
# También se extrae la longitud del ciclo                                  
   def Listaytipo(self):
        d1={};
        i=0;
        d1.update({"offset":float(self.sem.get("offset"))})
        self.ciclo=0
        self.programa=[]
        self.programacambios=[]
        for fase in self.sem.iter('phase'):
            self.ciclo+=float(fase.get("duration"))
            if float(fase.get("duration"))>9:
                d1.update({fase.get('state'):float(fase.get("duration"))})
                self.programacambios.append(i)
                           
            i=i+1
        return d1        
   def Prog(self):
        i=0
        for fase in self.sem.iter('phase'):
            self.programa.append({fase.get('state'):float(fase.get("duration"))})
           

        

##  Extraccion de fases vecinas y propias.
   def L(self,root):
        self.fases=[]
        self.fases.append(float(self.sem.get("offset")))
        for fase in self.sem.iter('phase'):
            if float(fase.get("duration"))>9:
                self.fases.append(float(fase.get("duration")))
        for i in self.vinc:
            a='./tlLogic[@id="'+i+'"]'
            a1=root.findall(a)[0]
            self.fases.append(float(a1.get("offset")))
            for fase in a1.iter('phase'):
                if float(fase.get("duration"))>9:
                    self.fases.append(float(fase.get("duration")))
## Extraccion de acciones posibles. Estas vienen dadas en forma de lista, donde cada entrada es un diccionario, el cual contine 
# suma que se le tiene que hacer a cada fase, menos a la ultima.
   def Acciones(self,margen):
       dic1={}
       a=self.Listaytipo()
       del a[self.ultimoestado]
       for p in margen:
           l=0
           m=0
           o=l
           op=pow(5,len(a))-1
           opb3=len(np.base_repr(op,base=5))
           b1={}
           v=[]
           lacc=list(a)
           for i in range(op+1):
               k=np.base_repr(i,base=5,padding=opb3-len(np.base_repr(i,base=5))+2)[::-1]
               for j in range(opb3):
                   b1.update({lacc[j]:[-int(p),-int(p)/2,0,int(p)/2,p][int(k[j])]  })           
               v.append(b1)
               b1={}
           dic1.update({str(int(p)):v})
           v=[]
           
       self.accionred=dic1 
       # v=list(self.accio)
       # for i in range(op+1):
           
       #     k=np.base_repr(i,base=3,padding=opb3-len(np.base_repr(i,base=3)))
       #     for j in range(opb3):
       #         b.update({v[j]:[-0.5,0,0.5][int(k[j])]  })
               
            
       #     self.acciored.update({i:b})
       #     b={}

           
## Cambiar el estado. Primero z                
   def Efectuaraccion(self,margen,rate):
       x=0
       a=[j for i,j in self.accionelegida.items() if i!="offset"]
       
       if all(i >= 10 for i in a):
           x=0
           for i in self.accionelegida:
               self.estado[i]=self.accionelegida[i]
               
          

       else:
            self.select_action(margen,1)
            self.Efectuaraccion(margen,1)
       
            # Posible bug numeros negativos.
   def select_action(self,margen,rate):
       listaestados=list(self.Listaytipo())[1:]
       random.shuffle(listaestados)
       destados={}
       sumaestados=sum([j for i,j in self.estado.items() if i!="offset"])
       k=0
       v=list(range(10,int(sumaestados/len(listaestados)*2*.75),10))
       k=0
       if len(listaestados)>1:
           for estado in listaestados[0:-1]:
               estadonuevo=random.choice(v)
               destados.update({estado:estadonuevo})
               k+=estadonuevo
       destados.update({listaestados[-1]:sumaestados-k})
       destados.update({"offset": random.choice(list(range(0,85,10)))})
       self.accionelegida=destados
       
                               
           
           
       

   def inicio(self, strategy, num_actions, device):
       self.current_step = 0
       self.strategy = strategy
       self.num_actions = num_actions
       self.device = device
                 
   def get_options():
      optParser = optparse.OptionParser()
      optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
      options, args = optParser.parse_args()
      return options   
    
       
   def Carreteras(self,dedges):
       self.aristas=[]
       for i in self.vinc:
           self.aristas.append(dedges.at[dedges.loc[(dedges["from"]==i)&(dedges["to"]==self.nombre)].index[0],"id"])
           
   def Tiempo_espera_interseccion(self):
       for i in self.aristas:
           self.tiempoespera+=traci.edge.getWaitingTime(i)
           
           
   def Recompensa(self,Sems):
       self.recom=-self.tiempoespera
       for i in self.vinc:
           self.recom+=-Sems[i].tiempoespera/len(self.vinc)
       for i in self.vout:
           self.recom+=-Sems[i].tiempoespera/len(self.vout)
       return self.recom
# %%   
def Extraccion(nombre_mapa): 
    dom=ET.parse(nombre_mapa)
    root = dom.getroot()
    df=pd.DataFrame(columns=['id','x','y'])
    junction=pd.DataFrame(columns=['id','x','y'])
    ## Junctions
    v=root.findall('./junction[@shape]')
    for elm in v:
        junction=junction.append({'id': elm.attrib['id'],'x':elm.attrib['x'],'y':elm.attrib['y']}, ignore_index=True)
    
    ## Traffic light
    v=root.findall('./junction[@type="traffic_light"]')
    for elm in v:
        df=df.append({'id': elm.attrib['id'],'x':elm.attrib['x'],'y':elm.attrib['y']}, ignore_index=True)
    df_tl=df.copy()
    for elm in root.findall('./edge[@to]'):
        if df_tl.loc[df_tl['id'] == elm.attrib["to"]].empty  == False:
            df=df.append({'id': elm.attrib["from"]}, ignore_index=True)
    df=df.drop_duplicates(subset=['id'], keep='first')
    df=df.reset_index()
    
    
    
    m=df_tl.shape[0]
    n=df.shape[0]
    Matriz_semaforos = csr_matrix((m,m ),dtype = np.int8).toarray()
    Matriz_semaforos=coo_matrix(Matriz_semaforos)
    Matriz_semaforos=Matriz_semaforos.tocsr()
    
    for elm in root.findall('./edge[@to]'):
        if df_tl.loc[df_tl['id'] == elm.attrib["from"]].empty == False and df_tl.loc[df_tl['id'] == elm.attrib["to"]].empty==False:
    
            Matriz_semaforos[ df_tl.loc[df_tl['id']==elm.attrib["from"]].index[0],df_tl.loc[df_tl['id']==elm.attrib["to"]].index[0]]=1
            # js[df_tl.loc[df_tl.loc[df_tl['id']==elm.attrib["from"]].index[0], df_tl['id']==elm.attrib["to"]].index[0]]=len(elm.getchildren())
            # jl[df_tl.loc[df_tl.loc[df_tl['id']==elm.attrib["from"]].index[0], df_tl['id']==elm.attrib["to"]].index[0]]=float(elm.getchildren()[0].attrib['length'])
            # jv[df_tl.loc[df_tl.loc[df_tl['id']==elm.attrib["from"]].index[0], df_tl['id']==elm.attrib["to"]].index[0]]=float(elm.getchildren()[0].attrib['speed'])
            # jc[df_tl.loc[df_tl.loc[df_tl['id']==elm.attrib["from"]].index[0], df_tl['id']==elm.attrib["to"]].index[0]]=len(elm.getchildren())
    
    ##Añadir x, y
    for i in list(df.id):
        df.loc[df['id']==i,'x':'y']=junction.loc[junction['id']==i,'x'].iloc[0],junction.loc[junction['id']==i,'y'].iloc[0]
    ## Matriz de Conexion de intersecciones
    JM = csr_matrix((n, n),dtype = np.int8).toarray()
    ## Matriz de conexion con numero de carriles
    JC = csr_matrix((n, n),dtype = np.int8).toarray()
    ## Matriz de posicion de semaforo en la intersecciones
    JS = csr_matrix((n, n),dtype = np.int8).toarray()
    ## Matriz de longitud de segmento
    JL = csr_matrix((n, n),dtype = np.float).toarray()
    ## Matriz de velocidad
    JV = csr_matrix((n, n),dtype = np.float).toarray()
    ## Matriz de velocidad
    JI = csr_matrix((n, n),dtype = np.float).toarray()
    JM=coo_matrix(JM)
    JC=coo_matrix(JC)
    JS=coo_matrix(JS)
    JL=coo_matrix(JL)
    JV=coo_matrix(JV)
    JI=coo_matrix(JI)
    jm=JM.tocsr()
    js=JS.tocsr()
    jl=JL.tocsr()
    jv=JV.tocsr()
    ji=JI.tocsr()
    jc=JC.tocsr()
    dedges=pd.DataFrame(columns=['id','from','to','index'])
    for elm in root.findall('./edge[@from]'):
        dedges=dedges.append({'id': elm.attrib["id"],'from':elm.attrib["from"],'to':elm.attrib["to"],'index':[]}, ignore_index=True)
    for elm in root.findall('./edge[@to]'):
        if df.loc[df['id'] == elm.attrib["from"]].empty == False and df.loc[df['id'] == elm.attrib["to"]].empty==False:
    
            jm[ df.loc[df['id']==elm.attrib["from"]].index[0],df.loc[df['id']==elm.attrib["to"]].index[0]]=1
            # js[df.loc[df.loc[df['id']==elm.attrib["from"]].index[0], df['id']==elm.attrib["to"]].index[0]]=len(elm.getchildren())
            # jl[df.loc[df.loc[df['id']==elm.attrib["from"]].index[0], df['id']==elm.attrib["to"]].index[0]]=float(elm.getchildren()[0].attrib['length'])
            # jv[df.loc[df.loc[df['id']==elm.attrib["from"]].index[0], df['id']==elm.attrib["to"]].index[0]]=float(elm.getchildren()[0].attrib['speed'])
            # jc[df.loc[df.loc[df['id']==elm.attrib["from"]].index[0], df['id']==elm.attrib["to"]].index[0]]=len(elm.getchildren())
    
    for elm in root.findall('./connection[@tl]'):
        dedges.iloc[dedges.iloc[(dedges['id'] == elm.attrib["from"]).values,[3]].index[0]][3].append(elm.attrib["linkIndex"])

    dedges.iloc[1][3].append(1)
    def inverse_mapping(f):
        return f.__class__(map(reversed, f.items()))
    
    G=nx.from_scipy_sparse_matrix(Matriz_semaforos, parallel_edges=False, create_using=nx.DiGraph)
    listcoloring=nx.coloring.greedy_color(G)
    b=nx.coloring.greedy_color(G)
    modul1=greedy_modularity_communities(G.to_undirected())
    modul2=[]
    for i in modul1:
        modul2.append(list(i))
    for k in range(len(modul2)):
        for i in range(len(modul2[k])):
            modul2[k][i]=df_tl.iloc[i][0]
            
    for k in b:
        listcoloring[df_tl.iloc[k][0]]=listcoloring.pop(k)
    
    coloringlist = {}
    for k, v in listcoloring.items():
        coloringlist[v] = coloringlist.get(v, []) + [k]
    return jm,df,df_tl,root,dedges,coloringlist,modul2

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
        math.exp(-1. * current_step * self.decay)
    

def Accionrama(rama,alpha,hoja,episode,timestep,Sems,Paccion,dom,k,coloringlist):
    rate=1
    if hoja==0 and episode==0 and timestep==0 and k==0:
        shutil.copy2("/dev/shm/Mapa_volumen"+str(alpha)+"rama"+str(rama)+".net.xml", "/dev/shm/Mapa_volumen"+str(alpha)+"rama"+str(rama)+"hoja"+str(hoja)+".net.xml")
    elif hoja==0:
        shutil.copy2("/dev/shm/Mapa_volumen"+str(alpha)+"rama"+str(rama)+".net.xml", "/dev/shm/Mapa_volumen"+str(alpha)+"rama"+str(rama)+"hoja"+str(hoja)+".net.xml")
    else:
        
        for i in coloringlist[k]:
            if 0.1>random.random():
                Sems[i].select_action(Paccion[0],rate)
                Sems[i].Efectuaraccion(Paccion[0],rate)
                Sems[i].sem.set("offset", str(Sems[i].estado["offset"]))
                for j in Sems[i].sem.iter("phase"):
                    if j.attrib["state"] in Sems[i].estado:
                        j.set("duration", str(Sems[i].estado[j.attrib["state"]]))
        dom.write("/dev/shm/Mapa_volumen"+str(alpha)+"rama"+str(rama)+"hoja"+str(hoja)+".net.xml")

def Accionmezclada(rama,alpha,hoja,episode,timestep,Sems,Paccion,dom,k,coloringlist,listaestado,prob):
    rate=1
    for i in coloringlist[k]:
        if prob/17<random.random():
            Sems[i].sem.set("offset", str(listaestado[i]["offset"]))
            for j in Sems[i].sem.iter("phase"):
                if j.attrib["state"] in listaestado[i]:
                    j.set("duration", str(listaestado[i][j.attrib["state"]]))
    dom.write("/dev/shm/Mapa_volumen"+str(alpha)+"rama"+"prob"+str(prob)+".net.xml")

def sumo(conection,alpha,hoja,rama,df_tl, Sems,coloringlist,k,file_name):
    # print("Hello")
    # time.sleep(1)
    a={}
    traci.start(["sumo", "-c", file_name,'--no-warnings',   "--no-step-log",
                                  "--tripinfo-output", "tripinfo.xml"],label=conection)
    con=traci.getConnection(conection);
    step = 0
    a={}
    while 4000> step:
        con.simulationStep();
        step += 1
        for i in df_tl["id"]:
            Sems[i].Tiempo_espera_interseccion();

    traci.switch(conection)
    traci.close()
    b=0
    for i in Sems:
        b+=Sems[i].tiempoespera
    return b;
def helper(n):
    return sumo(n[0], n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8],);


 
def argmax(lst):
  return lst.index(max(lst))

# %%
def Evaluacion(mapass,df_tl,Sems,nombre_config,jm,df,dedges,coloringlist,Rutas,Adds,flujo):
    data=[]
    for mapa in mapass:
        
        for flu in flujo:
            label=[]
            values=[]
            nombre_conf='/home/jj/Documents/tfm/final/Secuenciaciones/sim_volumenmapa'+str(mapa)+'flujo'+str(flu)+'.sumocfg'
            for i in range(10):
                label.append(("conection"+str(mapa)+str(flu)+str(i),i,flu,flu,df_tl, Sems,coloringlist,flu,nombre_conf))
                        
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in executor.map(helper, label):
                    values.append(result)
            res=[]
            for i in values:
                res.append(i)
            data.append([mapa, flu, mean(res)])

    df = pd.DataFrame(data, columns=['Configuaracion', 'Flujo', 'Tiempo de espera'])            
    for mapa in mappas:
        ax=df.plot(x='Flujo',y='Tiempo de espera',kind='hist')
        ax.figure.savefig('/Documents/Latex tfm/LAtex/Imagenes/Practica/Resultados/fijos/'+mapa+'.png')

    
   
def Rutass(prob,nombre_mapa,nombre_ruta,nombre_add,nombre_ruta_nueva,nombre_ruta_nueva_add):
    dicc=Extraccionmapa(nombre_mapa, nombre_ruta,nombre_ruta_nueva)
    extraccionruta(dicc,prob,nombre_add,nombre_ruta_nueva_add)
# %%

def main(alpha):
    nombre_mapa="/home/jj/Documents/tfm/da/rutaexper.net.xml"
    nombre_ruta='/home/jj/Documents/tfm/da/trips.trips.xml'
    nombre_config="/home/jj/Documents/tfm/mapasyrutas/sumo.sumocfg"
    additional = ET.Element("additional")
    jm,df,df_tl,root,dedges,coloringlist,modul2=Extraccion(nombre_mapa)
    domcfg=ET.parse(nombre_config)
    rootcfg = domcfg.getroot()
    Sems={}
    Paccion=[10.21]
    for i in df_tl["id"]:
        a='./tlLogic[@id="'+i+'"]'
        Sems.update({i:Extr(root.findall(a)[0],Paccion,jm,df,df_tl,root,dedges)})

    domm=ET.parse(nombre_mapa)
    rootm = domm.getroot()
    v=rootm.findall('./tlLogic')
    for tl in v:
        tl.set('programID', 'accion0')
        additional.append(tl)
        dicfases=[]
        fases=[]
        for i in tl.findall("./phase"):
            if float(i.attrib["duration"])>10:
                dicfases.append({"duration":i.attrib["duration"],"state":i.attrib["state"]})
                fases.append(i.attrib["state"])
        l1=[]
        for k in range(len(dicfases)):
            l1.append(copy.deepcopy(tl))
            l1[k].set('programID', 'accion'+str(k+1))
            for i in l1[k].findall('./phase'):
                if dicfases[k]["state"]==i.get("state"):
                    i.set("duration",str(float(dicfases[k]["duration"])+10))
                elif i.get("state") in fases:
                    i.set("duration",str(float(dicfases[k]["duration"])-10/(len(dicfases)-1)))
        
            additional.append(l1[k])           



    s=ET.ElementTree(additional)
    s.write("/home/jj/Documents/tfm/da/mapas/programa.add.xml")


    for i in rootcfg.iter("net-file"):
        i.set("value", nombre_mapa)
    for i in rootcfg.iter("additional-files"):
        i.set("value", nombre_ruta+" , /home/jj/Documents/tfm/da/mapas/programa.add.xml")
    domcfg.write('/home/jj/Documents/tfm/da/mapas/dqn.sumocfg')



           
if __name__=="__main__":
    main(1)