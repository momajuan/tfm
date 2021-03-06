"""
Created on Sun Jan  2 18:35:30 2022

@author: jj
"""
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
class Target(nn.Module):
       
   def __init__(self,m,n,device,numacc):
       super().__init__()
       self.device="cuda"
       self.memory=ReplayMemory(10000)
       self.numacc=numacc
       self.m=m
       c1=1028
       c2=512
       c3=256
       ## Aproximador
       self.fc1 = nn.Linear(in_features=m, out_features=c1)  
       self.fc2 = nn.Linear(in_features=c1, out_features=c2)
       self.fc3 = nn.Linear(in_features=c2, out_features=c3)

       self.out = nn.Linear(in_features=c3, out_features=numacc)
       self.optimizer = optim.Adam(params=self.parameters(), lr=lr) 
       self.loss=0
     #     self.estado.append(selfsita)
       # in_features=


   def forward(self, t):
       t = t.flatten(start_dim=1)
       t = F.relu(self.fc1(t))
       t = F.relu(self.fc2(t))
       t = F.relu(self.fc3(t))

       t = self.out(t)     
       return t  
# %%   
class Extr():
    
   def __init__(self,sem,jm,df,df_tl,root,dedges,numacc,edges,Estedges):
       self.edges=edges
       self.numacc=numacc
       self.accionelegida=random.choice(range(4))
       self.traficointerseccion=0
       self.sem=sem #Pasamos la interseccion a la clase
       self.nombre=self.sem.attrib["id"]
       self.vinc=[] # Vecinos incidentes
       for i in jm[:,df.loc[df['id']==self.sem.attrib['id']].index[0]].nonzero()[0]:
           if i<df_tl.shape[0]:
               self.vinc.append(df_tl.iloc[i][0])
               for j in jm[:,df.loc[df['id']==df_tl.iloc[i][0]].index[0]].nonzero()[0]:
                   if j<df_tl.shape[0]:
                       self.vinc.append(df_tl.iloc[i][0])
                            
       self.vout=[] # Vecinos incidentes
       self.acciones=[]
       for i in jm[df.loc[df['id']==self.sem.attrib['id']].index[0],:].nonzero()[1]:
           if i<df_tl.shape[0]:
               self.vout.append(df_tl.iloc[i][0])
               for j in jm[:,df.loc[df['id']==df_tl.iloc[i][0]].index[0]].nonzero()[0]:
                   if j<df_tl.shape[0]:
                       self.vout.append(df_tl.iloc[i][0])
       self.interentrada=set(self.vout+self.vinc)
       self.trafico=[0 for i in range(len(self.interentrada)+1)]
           
       ## Aproximador
       self.entradared=3*len(edges[self.nombre])+len(self.interentrada)+1
       c1=34
       c2=64
       self.cout=4 #numero de acciones

       
       self.tiempoespera=0
       self.tiempoesperares=0
       self.tecoloracion={}
       self.Carreteras(dedges)
       self.memory=ReplayMemory(10000)       #     self.estado.append(selfsita)
       # in_features=


        



## Extraccion de acciones posibles. Estas vienen dadas en forma de lista, donde cada entrada es un diccionario, el cual contine 
# suma que se le tiene que hacer a cada fase, menos a la ultima.
   def select_action(self,red,rate,Estado,Estedges,Redes, tru=True,evaluacionred=False,anterior=False):
       if tru==True:
           self.fasesantiguas=self.fases
           self.accionant=self.accionelegida
           self.fases=[self.accionelegida]
           for ested in Estedges:
               for i in self.edges[self.nombre]:
                   self.fases+=[ested[i]['trafico']]       
           for i in self.interentrada:
               self.fases+=[Estado[i]['accion']]
           if anterior==True:
               
               return Estado
           if evaluacionred==True:
               with torch.no_grad(): 
                     self.accionelegida=int((Redes[red][self.nombre].forward(torch.FloatTensor([self.fases]).to(device))+Redes[(red+1)%2][self.nombre].forward(torch.FloatTensor([self.fases]).to(device))).argmax()) # exploit 
                    #self.accionelegida=0 
                   # Redes[red][self.nombre].forward(torch.FloatTensor([self.fases]).to(device)+Redes[red][self.nombre].forward(torch.FloatTensor([self.fases]).to(device)
               Estado[self.nombre].update({'accion':self.accionelegida})
               return Estado   
           if rate > random.random(): 
             
               self.accionelegida=random.choice(range(self.numacc))
               Estado[self.nombre].update({'accion':self.accionelegida})
               return Estado# explore   
               # time.sleep(2) 
           else: 
               with torch.no_grad(): 
                   self.accionelegida=int(Redes[red][self.nombre].forward(torch.FloatTensor([self.fases]).to(device)).argmax()) # exploit 
                
               Estado[self.nombre].update({'accion':self.accionelegida})
               return Estado    #self.accionelegida=self.acciored[int(policy_net(state).argmax(dim=1).to(self.device)[0])] # exploit 
           
       else:

           self.fases=[self.accionelegida,Estado[self.nombre]['trafico']]
           for ested in Estedges:
               for i in self.edges[self.nombre]:
                   self.fases+=[ested[i]['trafico']]       
           for i in self.interentrada:
               self.fases+=[Estado[i]['accion']]
            
           if 1 > random.random(): 
             
               self.accionelegida=0
               Estado[self.nombre].update({'accion':self.accionelegida})
               return Estado# explore   
               # time.sleep(2) 
           
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
           self.tiempoesperares+=traci.edge.getWaitingTime(i)
           
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
    
    ##A??adir x, y
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
    listcoloring=nx.coloring.greedy_color(G,strategy="largest_first")
    b=nx.coloring.greedy_color(G,strategy="largest_first")
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
       
def sumo(conection,df_tl, Sems,coloringlist,file_name,Estado,r1,r2,Estadoedges,Estedges,Redes,t):
    print(coloringlist)
    print("holaaa")
    Experience = namedtuple(     'Experience',     ('state', 'action', 'next_state', 'reward') ) 
    # print("Hello")
    # time.sleep(1)
    AnteEstadoEdges=copy.deepcopy(Estadoedges)
    
    Ant2Estadoedges=copy.deepcopy(Estadoedges)

    eps_start = 1
    eps_end = 0.03 
    eps_decay = 0.002

    num_episodes = 10000 
     
    getrate=EpsilonGreedyStrategy(eps_start,eps_end,eps_decay) 
    
    a={}
    traci.start(["sumo", "-c", file_name,'--no-warnings',   "--no-step-log",
                                  "--tripinfo-output", "tripinfo.xml"],label=conection)
    con=traci.getConnection(conection);
    for j in Sems:
        con.trafficlight.setProgram(j,'accion'+str(0))
        # con.trafficlight.setProgram(j,'accion'+str(Sems[j].accionelegida))

    step = 0
    a={}
    Estedges={}
    AntEstadoEdges={}
    Ant2Estadoedges={}
    diccfases={}
    for j in Sems:
        diccfases[j]=[]
    for i in Estadoedges:
        Estadoedges.update({i:{"trafico":0}})
    AntEstadoEdges=copy.deepcopy(Estadoedges)
    Ant2Estadoedges=copy.deepcopy(Estadoedges) 
    Estedges=[Estadoedges,AntEstadoEdges,Ant2Estadoedges]
    Ant2Estadoedges=copy.deepcopy(AntEstadoEdges)
    AntEstadoEdges=copy.deepcopy(Estadoedges)
    Estedges=Estedges=[Estadoedges,AntEstadoEdges,Ant2Estadoedges]
    Estado={}
    for i in Sems:
        Estado.update({i:{'trafico':0,'accion':Sems[i].accionelegida}})
    for j in Sems:
        Estado=Sems[j].select_action(0,0,Estado,Estedges,Redes,"Hola")
    for j in Sems:
        Estado=Sems[j].select_action(0,0,Estado,Estedges,Redes,"Hola")
    for j in Sems:
        Estado=Sems[j].select_action(0,0,Estado,Estedges,Redes,"Hola")
    contadorciclos=0
    listaexperiencias=[]
    while step<4500 or con.vehicle.getIDCount()!=0:
    # while step<5000:

    # while step<450:
        if step%1050 in range(1):
            i="gneJ21"
            print(Sems[i].accionelegida,con.trafficlight.getProgram(i))
        if step in range(90*4):
            rate=1
        else:
            rate=0
        # rate=0

    
        if step%90==0 and step!=0:
            # red=random.choice([0,1])
            red=0
            for j in Sems:
                Estado[j].update({'trafico':Sems[j].tiempoespera})
            # for j in coloringlist[contadorciclos%len(coloringlist)]:
            #     Estado=Sems[j].select_action(red,rate,Estado,Estedges,Redes)
            #     con.trafficlight.setProgram(j,'accion'+str(Sems[j].accionelegida))
                
            
            for color in coloringlist:
                # if color!=contadorciclos%len(coloringlist):
                    for j in coloringlist[color]:
                        Estado=Sems[j].select_action(red,rate,Estado,Estedges,Redes,evaluacionred=True)
                        con.trafficlight.setProgram(j,'accion'+str(Sems[j].accionelegida))
                        diccfases[j].append(Sems[j].accionelegida)
            for j in Sems:
                listaexperiencias.append([red,j,[[Sems[j].fasesantiguas], [Sems[j].accionant], [Sems[j].fases],[Sems[j].Recompensa(Sems)]]])
            
            contadorciclos+=1  
            for i in Sems:
                Sems[i].tiempoespera=0
            Estedges=[Estadoedges,AntEstadoEdges,Ant2Estadoedges]
            Ant2Estadoedges={}
            Ant2Estadoedges=copy.deepcopy(AntEstadoEdges)
            AntEstadoEdges={}
            AntEstadoEdges=copy.deepcopy(Estadoedges)
            Estadoedges={}

            for j in Ant2Estadoedges:
                Estadoedges.update({j:{"trafico":0}})
            
        # if step%90==0 and step!=0:
        #     # red=random.choice([0,1])
        #     red=0
        #     for j in Sems:
        #         Estado[j].update({'trafico':Sems[j].tiempoespera})
        #     # for j in coloringlist[contadorciclos%len(coloringlist)]:
        #     #     Estado=Sems[j].select_action(red,rate,Estado,Estedges,Redes)
        #     #     con.trafficlight.setProgram(j,'accion'+str(Sems[j].accionelegida))
                
            
        #     for color in coloringlist:
        #         # if color!=contadorciclos%len(coloringlist):
        #             for j in coloringlist[color]:
        #                 Estado=Sems[j].select_action(red,rate,Estado,Estedges,Redes,evaluacionred=True)
        #                 con.trafficlight.setProgram(j,'accion'+str(Sems[j].accionelegida))
        #                 diccfases[j].append(Sems[j].accionelegida)
        #     for j in Sems:
        #         listaexperiencias.append([red,j,[[Sems[j].fasesantiguas], [Sems[j].accionant], [Sems[j].fases],[Sems[j].Recompensa(Sems)]]])
            
        #     contadorciclos+=1  
        #     for i in Sems:
        #         Sems[i].tiempoespera=0
        #     Estedges=[Estadoedges,AntEstadoEdges,Ant2Estadoedges]
        #     Ant2Estadoedges={}
        #     Ant2Estadoedges=copy.deepcopy(AntEstadoEdges)
        #     AntEstadoEdges={}
        #     AntEstadoEdges=copy.deepcopy(Estadoedges)
        #     Estadoedges={}

        #     for j in Ant2Estadoedges:
        #         Estadoedges.update({j:{"trafico":0}})
            
        con.simulationStep();
        for j in Estadoedges:
            Estadoedges[j].update({"trafico":copy.deepcopy(Estadoedges[j]["trafico"])+con.edge.getLastStepVehicleNumber(j)})
  
        step += 1
        for i in df_tl["id"]:
            Sems[i].Tiempo_espera_interseccion();




    traci.switch(conection)
    traci.close()
    b=0
    dicctiempoespera={}
    for i in Sems:
        dicctiempoespera[i]=Sems[i].tiempoesperares
        b+=Sems[i].tiempoesperares
        Sems[i].tiempoesperares=0
        Sems[i].tiempoespera=0
        Sems[i].traficointerseccion=0
    print(step)
    Estado
    return [listaexperiencias,b,step,diccfases,dicctiempoespera];
def helper(n):
    return sumo(n[0], n[1],n[2],n[3],n[4],n[5],n[6],n[7],n[8],);
Experience = namedtuple(     'Experience',     ('state', 'action', 'next_state', 'reward') ) 
def extract_tensors(experiences): 
    # Convert batch of Experiences to Experience of batches 
    batch = Experience(*zip(*experiences)) 
 
    t1 = torch.cat(batch.state) 
    t2 = torch.cat(batch.action) 
    t3 = torch.cat(batch.reward) 
    t4 = torch.cat(batch.next_state) 
 
    return (t1,t2,t3,t4) 
class QValues(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    @staticmethod 
    def get_current(policy_net, states, actions): 
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1)) 
    @staticmethod         
    def get_next(target_net, next_states):                 
        final_state_locations = False 
        non_final_state_locations = (final_state_locations == False) 
        non_final_states = next_states[non_final_state_locations] 
        batch_size = next_states.shape[0] 
        values = torch.zeros(batch_size).to("cuda") 
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach() 
        return values     

 
def argmax(lst):
  return lst.index(max(lst))

def BuckledeAprendizaje(Sems,Redes,Targets,listaexperiencias,i,k,o):
    Experience = namedtuple(     'Experience',     ('state', 'action', 'next_state', 'reward') ) 
    batch_size = 1024
    target_update = 20 
    gamma=0.95
    contadorciclos=0
    for experiencia in listaexperiencias:
        contadorciclos=0
        red=experiencia[0]
        j=experiencia[1]
        exper=experiencia[2]
        Redes[red][j].memory.push(Experience(torch.FloatTensor(exper[0]).to("cuda"),torch.FloatTensor( exper[1]).type(torch.int64).to("cuda"), torch.FloatTensor(exper[2]).to("cuda"),torch.FloatTensor( exper[3]).to("cuda")))
    if i==3 and k%2 ==0 and o%3:
        for _ in range(50):
            for j in Sems:
                if Redes[red][j].memory.can_provide_sample(batch_size): 
                    experiences = Redes[red][j].memory.sample(batch_size) 
                    states, actions, rewards, next_states = extract_tensors(experiences) 
                    current_q_values = QValues.get_current(Redes[red][j], states, actions) 
                    next_q_values = QValues.get_next(Targets[(red+1)%2][j], next_states) 
                    target_q_values = (next_q_values * gamma) + rewards 
                    Redes[red][j].loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1)) 
                    Redes[red][j].optimizer.zero_grad() 
                    Redes[red][j].loss.backward() 
                    Redes[red][j].optimizer.step() 
                if (contadorciclos+1) % target_update == 0: 
                            Targets[red][j].load_state_dict(Redes[red][j].state_dict())   
# %% 
def main():
    nombre_mapa="/home/jj/Documents/tfm/da/rutaexper.net.xml"
    nombre_ruta='/home/jj/Documents/tfm/da/trips.trips.xml'
    nombre_config="/home/jj/Documents/tfm/mapasyrutas/sumo.sumocfg"
    additional = ET.Element("additional")
    jm,df,df_tl,root,dedges,coloringlist,modul2=Extraccion(nombre_mapa)
    domcfg=ET.parse(nombre_config)
    rootcfg = domcfg.getroot()

    domm=ET.parse(nombre_mapa)
    rootm = domm.getroot()
    v=rootm.findall('./tlLogic')
    tls={}
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
        tls.update({tl.attrib["id"]:k+2})



    # s=ET.ElementTree(additional)
    # s.write("/home/jj/Documents/tfm/da/mapas/programa.add.xml")


    for i in rootcfg.iter("net-file"):
        i.set("value", nombre_mapa)
    for i in rootcfg.iter("additional-files"):
        i.set("value", nombre_ruta+" , /home/jj/Documents/tfm/da/mapas/programa.add.xml")
    domcfg.write('/home/jj/Documents/tfm/da/mapas/dqn.sumocfg')

    nombre_mapa='/home/jj/Documents/tfm/da/rutaexper.net.xml'
    jm,df,df_tl,root,dedges,coloringlist,modul2=Extraccion(nombre_mapa)
    Sems={} 
    Tar1={} 
    Tar2={} 
    Red1={} 
    Red2={} 
    Redes={} 
    Targets={}
    Estadoedges={}
    for i in dedges["id"]:
        Estadoedges.update({i:{"trafico":0}})
    AntEstadoEdges=copy.deepcopy(Estadoedges)
    Ant2Estadoedges=copy.deepcopy(Estadoedges) 
    Estedges=[Estadoedges,AntEstadoEdges,Ant2Estadoedges]
    edgesvecinos={}
    for i in df_tl["id"]:
        edgesvecinos.update({i:set(dedges.loc[dedges["from"]==i]["id"])})
    vecinos={}
    for i in df_tl["id"]:
        vecinos.update({i:set(dedges.loc[dedges["from"]==i]["to"])})
    edgesvecinosvecinos={}
    for i in vecinos:
        a=set(edgesvecinos[i])
        for j in vecinos[i]:
            try:
                a=a|set(edgesvecinos[j])
            except:
                pass
        edgesvecinosvecinos.update({i:list(a)})
    
    for i in df_tl["id"]: 
        a='./tlLogic[@id="'+i+'"]' 
        Sems.update({i:Extr(root.findall(a)[0],jm,df,df_tl,root,dedges,tls[i],edgesvecinosvecinos,Estedges)})
        Tar1.update({i:Target(Sems[i].entradared,Sems[i].cout,"cuda",tls[i]).to("cuda")}) 
        Tar2.update({i:Target(Sems[i].entradared,Sems[i].cout,"cuda",tls[i]).to("cuda")})  
        Red1.update({i:Target(Sems[i].entradared,Sems[i].cout,"cuda",tls[i]).to("cuda")}) 
        Red2.update({i:Target(Sems[i].entradared,Sems[i].cout,"cuda",tls[i]).to("cuda")}) 

    Redes.update({0:Red1}) 
    Redes.update({1:Red2}) 
    Targets.update({0:Tar1}) 
    Targets.update({1:Tar2})
# RedesFinales/5000FINAL553paramgneJ19.pth
    for i in Redes: 
        for j in Redes[i]: 
            pass
    # if os.path.isfile("/home/jj/Documents/tfm/da/Redes/redexp"+str(0)+"param"+j+".pth" )==True:
    #     print("Todo bien")
    #     for i in Redes: 
    #         for j in Redes[i]: 
    #             Path="/home/jj/Documents/tfm/da/RedesFinales/5000FINAL16b"+str(26*8)+"param"+j+".pth" 
    #             # Path="/dev/shm/5000FINAL16bparam"+j+".pth"
    #             Redes[i][j].load_state_dict(torch.load(Path, map_location="cuda")) 
    # print(Path)
    Estado={}
    for i in Sems:
        Estado.update({i:{'trafico':0,'accion':Sems[i].accionelegida}})
    for i in Sems:
        Estado=Sems[i].select_action(1,1,Estado,Estedges,Redes,'h')
    data=[]
    steps=[]
    tempsespera=[]
    li=[1]
    # for m in range(int(1600//26//4)+1):
    #     for n in range(4): 
    #         li.append(26*m+n)
    for t in li:
        for k in range(1):
            for i in range(1):
                
                if os.path.isfile("/home/jj/Documents/tfm/da/Redes/redexp"+str(0)+"param"+j+".pth" )==True:
                    print("Todo bien")
                    for i in Redes: 
                        for j in Redes[i]: 
                            Path="/home/jj/Documents/tfm/da/RedesFinales/5000FINAL17mb"+str(52)+"param"+j+".pth" 
                            # Path="/dev/shm/5000FINAL16bparam"+j+".pth"
                            Redes[i][j].load_state_dict(torch.load(Path, map_location="cuda"))
                file_name='/home/jj/Documents/tfm/da/mapas/dqn.sumocfg'
    #            Redes,Targets,Sems,res=sumo('conection',df_tl, Sems,coloringlist,file_name,Estado,Redescpu,Targetscpu,i,k)
                label=[]
                values=[]
                l=0
                exper=sumo('conection'+str(l)+str(i)+str(k),df_tl, Sems,coloringlist,file_name,Estado,i,k,Estadoedges,Estedges,Redes,t)   
                o=0
    
                print("YA no estamos en aprendizaje")
                steps.append(exper[2])
                tempsespera.append(exper[1])
                print("Pasos de la simulaci??n ",steps)
                print("Tiempo espera",tempsespera)
                dicfases=exper[-2]
                dicte=exper[-1]
                diccam={}
                for i in dicfases:
                    o=0
                    for k in range(len(dicfases[i])-1):
                        if dicfases[i][k]!=dicfases[i][k+1]:
                            o+=1
                    diccam[i]=o
                print(diccam)
                # print(lo)
                # print(dicfases)
                # print(dicte)
            # for l in Redes: 
            #     for j in Redes[l]: 
            #         Path="/home/jj/Documents/tfm/da/Redes/red"+str(i)+"param"+j+".pth" 
            #         torch.save(Redes[l][j].state_dict(), Path)                         
                print(data)

if __name__=="__main__":
    main()
    