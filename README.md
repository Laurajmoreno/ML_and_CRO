# APLICACIÓN DE UN MODELO DE MACHINE LEARNING A LA ESTRATEGIA CRO DE UN E-COMMERCE

Gran parte de los esfuerzos de marketing en un e-commerce se centran en atraer tráfico a la web. Sin embargo, no siempre se aplican los mismos recursos para retener a esos usuarios y garantizar su conversión. Como respuesta a esta situación, en los últimos años, han surgido nuevas estrategias, en su mayoría manuales y rudimentarias en su puesta en práctica, dirigidas a **mejorar las tasas de conversión** de los sitios web.

La mayoría de estos procesos podrían **agilizarse, optimizarse y automatizarse** gracias a modelos de Aprendizaje Automático. Mediante el uso de un proceso generativo probabilístico se puede **modelizar el comportamiento del cliente** a tiempo real, **predecir su toma de decisiones** en contextos específicos y **adoptar medidas** adecuadas para favorecer su conversión.

Este proyecto busca establecer un **marco de trabajo inicial** para la implementación de un modelo de Machine Learning en el desarrollo de la estrategia de CRO de un e-commerce. El estudio se enmarca en el contexto de una tienda online al por menor, que opera fundamentalmente en el mercado español.

## INTRUCCIONES

### Documentación:

En este repositorio, se encuentran tanto los **notebooks** (numerados en el orden en el que deben ser explorados) como la **memoria** del proyecto (PROJECT_NARRATIVE.pdf). Asimismo, se incluye el fichero correspondiente al código fuente del *frontend* interactivo, que también puede ser sobrescrito y ejecutado en el último de los notebooks. 

Por otro lado, las **credenciales** de la cuenta de GMAIL necesaría para autenticarse en la API de *Google Analytics* y descargar los datos, han sido enviados a su correo electrónico. 

### Antes de empezar:

1.	Clone en local el repositorio de GIT HUB: https://github.com/Laurajmoreno/kschool_masterDS_TFM

2.	En el mismo directorio en el que haya creado el repositorio, cree una nueva carpeta llamada ***data***
	- En el conjunto de los notebooks, **data root** ha sido definido como `../data/`

3.	Conectese a la **cuenta de Gmail** que le ha sido facilitada por correo electrónico y abra la cuenta de **Google Drive** asociada.
	- Descargue y descomprima los ficheros de **‘data’** en el interior de la carpeta creada en el apartado anterior.
	- Descargue y descomprima los ficheros de **‘ficheros a incluir en el interior del repositorio’** en el interior del repositorio

4.	Cree el entorno Conda a partir del fichero **environment.yml** del repositorio y actívelo (todos los paquetes fueron instalados con *conda* y *conda-forge*, es posible que tarden en instalarse)
```
$ conda env create --prefix ./env --file environment.yml
$ conda activate ./env
```
		

5.	Ejecute **JUPYTER NOTEBOOK**

6.	Para facilitar la navegación en el interior de los notebooks, se recomienda activar **nbextensions** (ya instalada en el entorno) en la *Home* de *Jupyter Notebook*.


