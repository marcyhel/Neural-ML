# Biblioteca de redes neurais simples
Essa é uma iniciativa para que mais pessoas possam ter acesso as rede neurais, viabilizando prototipos rapidos e de facil compreenção 

## Instalação
A instalação pode ser feita atravez do gerenciador de pacodes do python utilizando o comando
``` python
pip install neuralml
```
## Importação
Para importar deve fazer a importação do pacode e chamar o codigo rede
```python
from neuralml import rede
```
## Iniciando a Rede e adicinando camadas de neuronios
Para que possamos iniciar a rede basta instancia a classe `RedeNeural()` é dentro dela que vamos fazer todas as configurações nessesarias para a nossa rede
```python
rede_neural = rede.RedeNeural()
```
As camadas são adicionadas pela função `addNeuronio` da seguinter forma, devemos pensar que estamos adicinando não as camadas e sim as ligaões entre elas se fizermo 
`addNeuronio(2,3)` estamos dizendo que nossa rede tem 2 neuronios na camada de entrada e 3 de saida, mas se continuarmos adicionando ligação entre camada então a camada de saida passa a ser a ultima a ser adicionada segue o exemplo:

```python
rede_neural.addNeuronio(2,5)
rede_neural.addNeuronio(5,4)
rede_neural.addNeuronio(4,1)
```
Nesse exemplo a nossa rede tem 2 neuronios na camada de entrada 5 na primeira camada oculta, 4 na segunda camada oculta e tem 1 neuronio na saida, lembrando a ligação com a 
a proximacada tem que iniciar com a mesma quantidade de ligação que o final da anterior

## Configurações adicionais
Algumas coisas que podemos configurar por enquanto é o learning rate de nossa rede da seguinte forma `rede_neural.addLearningRate(0.01) ` isso afeta o quanto vai ser o passo de 
aprendizado da rede a cada epoc
podemos alterar a função de ativação da seginte forma `rede_neural.ativador = rede.RedeNeural.tanh`
|Ativação||
|------|-----|
|sigmoid | Utilizado para calculos não lineares |
|tanh|Utilizado para calculos não lineares|

Breve será adicionados outras funções de ativação
### Lista de funções

|Funções| entrada|retorno|
|------|-----|----|
|predict|.predict(array)|Matriz|
|treinar|.treinar(array_entradas, array_saidas, epoc)||
|save|.save(nome = "Nome_do_arquivo")||
|open|.open(nome = "Nome_do_arquivo")||
|addNeuronio|.addNeuronio(2, 5)||
|addLearningRate|.addLearningRate(0.01)||
## Exemplos
### Exemplo 1
Treinando e salvando
```python
from neuralml import rede

redeneural = rede.RedeNeural()
redeneural.ativador = rede.RedeNeural.tanh

redeneural.addNeuronio(2,5)
redeneural.addNeuronio(5,1)

entrada = [[0,0],[0,1],[1,0],[1,1]]
saida = [[0],[0],[1],[1]]

redeneural.treinar(entrada,saida,epoc=6000)

print(redeneural.predict([1,1]))

rede_neural.save(nome="teste")
```
### Exemplo 2
Abrindo arquivos salvos
```python
from neuralml import rede

rede_neural = rede.RedeNeural()

rede_neural.ativador = rede.RedeNeural.tanh

rede_neural.addNeuronio(2,5)
rede_neural.addNeuronio(5,1)
rede_neural.open(nome="teste")

print(rede_neural.predict([1,1]))
```
