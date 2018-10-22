#https://www.analyticsvidhya.com/blog/2018/10/mining-online-reviews-topic-modeling-lda/
#Revisões de produtos on-line são uma ótima fonte de informações para os consumidores. Do ponto de vista dos vendedores,
# as avaliações on-line podem ser usadas para avaliar o feedback dos consumidores sobre os produtos ou serviços que estão vendendo.
# No entanto, uma vez que essas revisões on-line são muitas vezes esmagadoras em termos de números e informações, um sistema inteligente,
# capaz de encontrar informações importantes (tópicos) dessas revisões, será de grande ajuda tanto para os consumidores quanto para os vendedores.
# Este sistema servirá dois propósitos:

# 1 - Permitir que os consumidores extraiam rapidamente os principais tópicos cobertos pelas revisões sem ter que passar por todos elas.
# 2 - Ajudar os vendedores / varejistas a obter feedback do consumidor na forma de tópicos (extraídos das avaliações do consumidor).

#Para resolver essa tarefa, usaremos o conceito de Modelagem de Tópicos (LDA) nos dados da Amazon Automotive Review.
# Você pode baixá-lo neste link (snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz).
# Conjuntos de dados semelhantes para outras categorias de produtos podem ser encontrados aqui (http://jmcauley.ucsd.edu/data/amazon/).

#Por que você deve usar a modelagem de tópicos para esta tarefa?

#Como o nome sugere, Modelagem de Tópicos é um processo para identificar automaticamente os tópicos presentes em um objeto de texto e
# derivar padrões ocultos exibidos por um corpus de texto. Os modelos de tópicos são muito úteis para várias finalidades, incluindo:

# Document clustering
# Organizing large blocks of textual data
# Information retrieval from unstructured text
# Feature selection


# No nosso caso, em vez de documentos de texto, temos milhares de análises de produtos on-line para os itens listados na categoria "Automotivo".
# Nosso objetivo aqui é extrair um certo número de grupos de palavras importantes das avaliações.
# Esses grupos de palavras são basicamente os tópicos que ajudam a determinar o que os consumidores realmente estão falando nas avaliações.

# Aqui, trabalharemos na declaração de problema definida acima para extrair tópicos úteis do nosso conjunto de dados de avaliações on-line
# usando o conceito de Latent Dirichlet Allocation (LDA).