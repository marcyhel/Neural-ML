import numpy as np
import random
import copy
import math
import os
class Matriz:
	def __init__(self,linhas,colunas):
		self.linhas=linhas
		self.colunas=colunas
		self.dado=np.arange(float(self.linhas*self.colunas))
		self.dado.shape=(self.linhas,self.colunas)
		#self.aleatorizar()
	def zerar(self):
		for i in range(self.linhas):
			for j in range(self.colunas):
				self.dado[i][j]=0
	def limpa_num(self,num):
		for i in range(self.linhas):
			for j in range(self.colunas):
				self.dado[i][j]=num

	@staticmethod
	def soma(mat1,mat2):
		mat=Matriz(mat1.linhas,mat1.colunas)
		for i in range(mat1.linhas):
			for j in range(mat1.colunas):
				mat.dado[i][j]=mat1.dado[i][j]+mat2.dado[i][j]
		return mat
	@staticmethod
	def subtrair(mat1,mat2):
		mat=Matriz(mat1.linhas,mat1.colunas)
		for i in range(mat1.linhas):
			for j in range(mat1.colunas):
				mat.dado[i][j]=mat1.dado[i][j]-mat2.dado[i][j]
		return mat
	@staticmethod
	def hadamard(mat1,mat2):
		mat=Matriz(mat1.linhas,mat1.colunas)
		for i in range(mat1.linhas):
			for j in range(mat1.colunas):
				mat.dado[i][j]=mat1.dado[i][j]*mat2.dado[i][j]
		return mat
	@staticmethod
	def multiplica_escalar(mat1,const):
		mat=Matriz(mat1.linhas,mat1.colunas)
		for i in range(mat1.linhas):
			for j in range(mat1.colunas):
				mat.dado[i][j]=mat1.dado[i][j]*const
		return mat
	@staticmethod
	def transpor(mat1):
		mat=Matriz(mat1.colunas,mat1.linhas)
		mat.dado=copy.deepcopy(mat1.dado.T)
		return mat
	@staticmethod
	def multiplica(mat1,mat2):
		if mat1.colunas != mat2.linhas:
			print("não tem solucao")
		mat=Matriz(mat1.linhas,mat2.colunas)
		mat.zerar()
		"""Multiplica duas matrizes."""
		matrizR = copy.deepcopy(mat.dado)
		for i in range(mat1.linhas):
			for j in range(mat2.colunas):
				for k in range(mat1.colunas):
					# print(matrizR[i][j])
					mat.dado[i][j] += mat1.dado[i][k] * mat2.dado[k][j]
		return mat
	
	def aleatorizar(self):
		for i in range(self.linhas):
			for j in range(self.colunas):
				self.dado[i][j]=(random.random()*2)-1
	def constant(self,num):
		for i in range(self.linhas):
			for j in range(self.colunas):
				self.dado[i][j]=num
#2606

class RedeNeural:
	def __init__(self):
		self.neuronio_pronto=[]
		self.neuronio_recorre=[]
		self.pesos_recorre=[]
		self.neuronios=[]
		self.bias=[]
		self.numbias=0.5
		self.ativador=RedeNeural.sigmoid
		self.learning_rate=0.01
	
	@staticmethod
	def deriva_sigmoid(x):
		return RedeNeural.sigmoid(x) * (1-RedeNeural.sigmoid(x) )
	@staticmethod
	def deriva_tanh(x):
		return  1-math.tanh(x)**2
	@staticmethod
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))
	@staticmethod
	def tanh(x):
		return math.tanh(x)
	@staticmethod
	def add(x):
		return x+1
	def salvar(self,nome,mat):
		with open(nome+'.txt', 'w') as arquivo:
			for i in range(len(mat)):
				for j in range(len(mat[0])):
					arquivo.write(str(mat[i][j])+' ')
				arquivo.write('\n')
	def ler(self,nome):

		arquivo = open(nome+'.txt','r')

		texto = []  #declaro um vetor
		matriz = [] #declaro um segundo vetor
		texto = arquivo.readlines() #quebra as linhas do arquivo em vetores 
		

		for i in range(len(texto)):          #esse for percorre a posições dp vetor texto
			matriz.append(texto[i].split())  #aqui eu quebro nos espasos das palavras

		for x in range(len(matriz)):
			for i in range(len(matriz[0])):
				matriz[x][i]=float(matriz[x][i])
				#print(matriz[0][0])
		
		return matriz
	def addNeuronio(self,num1,num2):
		neu=Matriz(num2,num1)
		neu.aleatorizar()
		bias=Matriz(num2,1)
		bias.constant(self.numbias)
		recorre=Matriz(num2,num2)
		recorre.aleatorizar()
		self.bias.append(bias)
		self.neuronios.append(neu)
		self.pesos_recorre.append(recorre)
	def limparRecorre(self):
		self.neuronio_recorre=[]
	def map_deriva(self,mat):
		for i in range(mat.linhas):
			for j in range(mat.colunas):
				if(self.ativador==RedeNeural.sigmoid):
					mat.dado[i][j]=RedeNeural.deriva_sigmoid(mat.dado[i][j])
				elif(self.ativador==RedeNeural.tanh):
					mat.dado[i][j]=RedeNeural.deriva_sigmoid(mat.dado[i][j])
		return mat
	def map_ativar(self,mat):
		for i in range(mat.linhas):
			for j in range(mat.colunas):
				mat.dado[i][j]=self.ativador(mat.dado[i][j])
		return mat
	def addBias(self,num):
		self.numbias=num
	def addLearningRate(self,num):
		self.learning_rate=num
	def predict(self,arr):
		aux=0
		self.neuronio_pronto=[]
		for i in range(len(self.neuronios)):
			if(i==0):
				entrada=np.array(arr)
				entrada.shape=(len(arr),1)
				mat=Matriz(len(arr),1)
				mat.dado=entrada
				
				aux=Matriz.multiplica(self.neuronios[i],mat)
				
				aux=Matriz.soma(aux,self.bias[i])
				
				aux=self.map_ativar(aux)
				
			   
			else:
				
				aux=Matriz.multiplica(self.neuronios[i],aux)
				aux=Matriz.soma(aux,self.bias[i])
				aux=self.map_ativar(aux)
			
			self.neuronio_pronto.append(aux)
			
			
		return aux

	def predictRecore(self,arr):
		aux=0
		self.neuronio_pronto=[]
		for i in range(len(self.neuronios)):
			if(i==0):
				entrada=np.array(arr)
				entrada.shape=(len(arr),1)
				mat=Matriz(len(arr),1)
				mat.dado=entrada
				
				aux=Matriz.multiplica(self.neuronios[i],mat)
				
				
				
			else:
				aux=Matriz.multiplica(self.neuronios[i],aux)
				

			#print("------{}".format(i))
			#print(aux.dado)
			if(i<len(self.neuronios)-1):
				try:
					#self.neuronio_recorre[i]=Matriz.multiplica_escalar(self.neuronio_recorre[i],2)
					#aux = Matriz.hadamard(self.neuronio_recorre[i],copy.deepcopy(aux)) 
					aux_recorre=Matriz.multiplica(self.pesos_recorre[i],self.neuronio_recorre[i])
					aux=Matriz.soma(aux_recorre,aux)
					
				except:
					
					self.neuronio_recorre.append(copy.deepcopy(aux))
					
					self.neuronio_recorre[i].limpa_num(1)

					#aux = Matriz.hadamard(self.neuronio_recorre[i],copy.deepcopy(aux)) 
					aux_recorre=Matriz.multiplica(self.pesos_recorre[i],self.neuronio_recorre[i])
					aux=Matriz.soma(aux_recorre,aux)
				

			
				self.neuronio_recorre[i]=copy.deepcopy(aux)
				#print(self.neuronio_recorre[i].dado)
			aux=Matriz.soma(aux,self.bias[i])
			aux=self.map_ativar(aux)
			self.neuronio_pronto.append(aux)
			
		#print("---------")
		return aux
	def open(self,nome='none'):
		for i in range(len(self.neuronios)):
			self.neuronios[i].dado=self.ler(nome+str(i))
		for i in range(len(self.bias)):
			self.bias[i].dado=self.ler(nome+"_bias"+str(i))
		
	def save(self,nome='none'):
		for i in range(len(self.neuronios)):
			self.salvar(nome+str(i),self.neuronios[i].dado)
		for i in range(len(self.bias)):
			self.salvar(nome+"_bias"+str(i),self.bias[i].dado)
		
	def treinar(self,entrada,esperado,epoc=10000):

		for i in range(epoc):
			for c in range(len(entrada)):
				self.treinamento(entrada[c],esperado[c])
	def treinamento(self,arr,esperado):
		gradiente=0
		neu_ante=0
		saida=0
		saida=self.predict(arr)

		aux=np.array(esperado)
		aux.shape=(len(esperado),1)
		espera=Matriz(len(esperado),1)
		espera.dado=aux
		
		saida_erro=Matriz.subtrair(espera,self.neuronio_pronto[len(self.neuronios)-1])
		d_saida=self.map_deriva(saida)
		for i in range(len(self.neuronios)-1,-1,-1):
			#print(i)
			if(i==len(self.neuronios)-1):
				#print('entrada')
				neu_ante=Matriz.transpor(self.neuronio_pronto[i-1])
				gradiente=Matriz.hadamard(d_saida,saida_erro)
				gradiente=Matriz.multiplica_escalar(gradiente,self.learning_rate)
			  
				self.bias[i]=Matriz.soma(self.bias[i],gradiente)
				delta=Matriz.multiplica(gradiente,neu_ante)
				
				self.neuronios[i]=Matriz.soma(self.neuronios[i],delta)
				

			else:
				#print('meio')
				transpo=Matriz.transpor(self.neuronios[i+1])
				saida_erro=Matriz.multiplica(transpo,saida_erro)
				d_saida=self.map_deriva(self.neuronio_pronto[i])
				neu_ante=Matriz.transpor(self.neuronio_pronto[i])
				gradiente=Matriz.hadamard(d_saida,saida_erro)
				gradiente=Matriz.multiplica_escalar(gradiente,self.learning_rate)

				self.bias[i]=Matriz.soma(self.bias[i],gradiente)
				delta=Matriz.multiplica(gradiente,neu_ante)
			
				self.neuronios[i]=Matriz.soma(self.neuronios[i],delta)

