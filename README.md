# Segmentação de Vídeos por Clusters

Projeto onde é possível visualizar a segmentação de cenas de vídeos por clusters de descritores de alta dimensionalidade.

## Instalação

Você deve possuir no mínimo a versão de 3.7 de [python](https://www.python.org/downloads/).

Utilize o gerenciador de dependências [pip](https://pip.pypa.io/en/stable/) para instalar os pacotes necessários para rodar este programa.

Na raíz do projeto faça os seguintes passos:

```bash
cd src/
pip install -r requirements.txt
```

## Uso

Para iniciar o programa, na raiz do projeto faça os seguintes passos:

```bash
cd src/
streamlit run run.py
```

O projeto vem com uma pasta em sua raíz chamada ```/data```, ali existe um vídeo no qual você já pode testar a ferramenta.

Ao abrir a página da ferramenta você verá o seguinte:

![image](https://drive.google.com/uc?export=view&id=1pX1VdQg_sR-RRpXK_CZ_TukkuS5jfPnt)

O que acontece é que ao abrir a ferramenta ela já entende que há um vídeo dentro da pasta ```/data``` e começou a extrair suas features com o extrator SlowFast, como indicado no menu à esquerda. 

Neste menu é possível ver em ordem:
* A opção de alterar a pasta na qual a ferramenta procura os vídeos
* O vídeo atualmente selecionado e que está sendo visualizado
* O extrator de features selecionado, ou seja, os clusters visualizados foram formados com as features extraídas com este extrator 
* As opções de clusterização que formaram os clusters visualizados

Além disso, conseguimos ver o vídeo selecionado, a linha do tempo segmentada pelos clusters formados e um gráfico de dispersão de cada segmento clusterizado. 

Importante mencionar que cada ponto nesse gráfico de dispersão representa um segmento de 32 frames do vídeo selecionado.

Para alterar a pasta de vídeos que a ferramenta interage basta selecionar a opção _Change Directory_ e colocar o caminho absoluto da sua pasta de interesse.

![image](https://drive.google.com/uc?export=view&id=1cSwkJtJTso41HSnacjt7Z-Qg3hdpkLvD)

Para alterar o extrator de features e também de qual extrator você está visualizando as features, basta selecionar no dropdown. Atualmente só existem duas opções [SlowFast](https://arxiv.org/abs/1812.03982) e [I3D](https://arxiv.org/abs/1705.07750).

Lembrando que após a primeira vez que alguma combinação de vídeo e extrator é feita, o processo fica mais rápido já que as features agora estão salvas na pasta dos seus vídeos e a ferramentas as acessa por lá. 

![image](https://drive.google.com/uc?export=view&id=1OP6i9fezQKW5Jr8Ofbo008TDGpt8vz3g)

Por último, mas não menos importante são as opções de clusterização. Nela podemos:
 * Selecionar qual algoritmo de clustering queremos utilizar([KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) ou [Agglomerative](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html))
 * Selecionar se queremos aplicar [positional encoding](https://arxiv.org/abs/1706.03762) nas features antes de clusterizá-las
 * Selecionar se queremos utilizar um algoritmo baseado na [métrica de silhueta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) para escolher o número ótimo de clusters
 * Selecionar o número de clusters que queremos

![image](https://drive.google.com/uc?export=view&id=1oln_ADkOXuwMX287ZsOIWUOdvqEiXMtw)

## Licença
[MIT](https://choosealicense.com/licenses/mit/)
