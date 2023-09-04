import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Dados de exemplo: avaliações de filmes e suas categorias (positivo ou negativo)
documentos = [
    ("Ótimo filme, adorei!", "positivo"),
    ("Que filme horrível, péssimo!", "negativo"),
    ("Gostei muito do filme.", "positivo"),
    ("Não recomendo, muito chato.", "negativo"),
    ("Amei", "positivo")
    # Adicione mais exemplos de avaliações e suas categorias aqui
    ("Muito bom!", "positivo")
    ("Muito ruim!", "negativo")
    ("Melhor filme que eu já assisti", "positivo")
    ("Odiei", "negativo")
    ("Filme muito bom! Recomendo", "positivo")
    ("Não gostei, o filme é ruim", "negativo")
    ("Muito interessante e divertido!", "positivo")
    ("Pior filme que eu já assisti", "negativo")
    ("Este é o meu filme favorito!", "positivo")
    ("Não gostei nada desse filme", "negativo")
    ("Eu adorei este filme!", "positivo")
    ("Sem graça, muito ruim", "negativo")
    ("Filme muito bom!", "positivo")
    ("Muito chato! Odiei!", "negativo" )
    ("Muito legal! Adorei!", "positivo")
    
]

# Embaralhe os dados
random.shuffle(documentos)

# Separe os dados em textos e categorias
textos = [texto for texto, categoria in documentos]
categorias = [categoria for texto, categoria in documentos]

# Vetorização dos textos usando a contagem de palavras (CountVectorizer)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# Treine um modelo de classificação (Naive Bayes, por exemplo)
modelo = MultinomialNB()
modelo.fit(X, categorias)

# Função para fazer previsões com base no texto inserido pelo usuário
def fazer_previsao(texto):
    texto_transformado = vectorizer.transform([texto])
    categoria = modelo.predict(texto_transformado)[0]
    return categoria

# Exemplo de classificação de texto com entrada do usuário
while True:
    entrada_usuario = input("Digite uma avaliação de filme (ou 'sair' para encerrar): ")
    if entrada_usuario.lower() == "sair":
        break
    previsao = fazer_previsao(entrada_usuario)
    print(f"Previsão: {previsao}")
