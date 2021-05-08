
# LIBRERÍAS:
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import csv
import pickle
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import shap

# DIRECTORIO RAIZ DE LOS FICHEROS DE DATOS:
data_root="../data/"

# MENU
menu=['Introducción', 'Maximizar las conversiones','¿Cómo funciona?']
choice=st.sidebar.radio('Menú',menu)

if choice=='Introducción':
    st.title('Introducción')
    st.subheader('¿Cómo mejorar el ratio de conversión de un ecommerce sin morir en el intento?')
    st.markdown('Gran parte de los esfuerzos de marketing en un e-commerce se centran en atraer tráfico a la web.\
    Sin embargo, no siempre se aplican los mismos recursos para retener a esos usuarios y garantizar su conversión.\
    Como respuesta a esta situación, en los últimos años, han surgido nuevas estrategias, en su mayoría manuales \
    y rudimentarias en su puesta en práctica, dirigidas a **mejorar las tasas de conversión** de los sitios web. ')
    st.markdown('La mayoría de estos procesos podrían **agilizarse, optimizarse y automatizarse** gracias a modelos \
    de Aprendizaje Automático. Mediante el uso de un proceso generativo probabilístico se puede **modelizar\
    el comportamiento del cliente** a tiempo real, **predecir su toma de decisiones** en contextos específicos \
    y **adoptar medidas** adecuadas para favorecer su conversión.')
    st.markdown('Mediante un **ejemplo práctico** aplicado a la farmacia online Galileo 61 ilustraremos\
    los beneficios de este tipo de sistemas. Empezaremos por ver cómo podemos **influir y maximizar las\
    conversiones** a partir de un modelo preentrenado de Machine Learning.')
    st.markdown('*Pase a la siguiente página en el menú de la izquierda >*')

if choice=='Maximizar las conversiones':
        
        # INPUT:
        
        st.title('Ejemplo de maximización de las conversiones')
        st.subheader('¿Cuánto podrían haber aumentado las conversiones de haber aplicado un\
        descuento específico en el momento y lugar adecuados?')
        st.markdown('Siga las instrucciones:')
        
        features=['ga:productSKU','Product_price','ga:pagePath','ga:landingPagePath','ga:city','ga:dateHourMinute','Web_Discount']
        
        try:
            uploaded_file=st.file_uploader('Suba el fichero *prueba_streamlit.csv* disponible \
            en el directorio de datos', type='csv')  
            
            @st.cache(suppress_st_warning=True)
            def create_df():
                df=pd.read_csv(uploaded_file)
                df=df[features]
                df['ga:productSKU']=df['ga:productSKU'].astype('str')
                df['ga:pagePath']=df['ga:pagePath'].astype('str')
                df['ga:pagePath'] = df['ga:pagePath'].apply(lambda x: x[:x.find("?pag")] if "?pag" in x else x)
                df['Detail_View']=df['ga:pagePath'].apply(lambda url: 1 if url[-5:]=='.html' else 0)
                df['ga:landingPagePath']=df['ga:landingPagePath'].astype('str')
                df['ga:landingPagePath'] = df['ga:landingPagePath'].apply(lambda x: x[:x.find("?pag")] if "?pag" in x else x)        
                df['ga:city']=df['ga:city'].astype('str')
                df['ga:dateHourMinute']=pd.to_datetime(df['ga:dateHourMinute'],format='%Y%m%d%H%M')
                df['dateTime_month']=df['ga:dateHourMinute'].dt.month
                df['month_sin']=np.sin((df.dateTime_month-1)*(2.*np.pi/12))
                df['month_cos']=np.cos((df.dateTime_month-1)*(2.*np.pi/12))
                return df
            
            df=create_df()
        
        except ValueError:
            st.error('Arrastre el fichero o presione el botón para abrir el explorador de archivos.')
        
        try:
            # PREPROCESS
            
            ## Custom Transformers
            
            class SelectColumnTransformer(BaseEstimator,TransformerMixin):
                def fit(self,X,y=None):
                    return self
                def transform (self,X,y=None):
                    return X
            
            def engineer_feature(columns, X):
                df = pd.DataFrame(X, columns=columns)
                df["Product_price"] = df["Product_price"]*(1-df["Web_Discount"])
                return df
                
            ## Transformation
            final_features= ['ga:productSKU','ga:city','ga:pagePath','ga:landingPagePath','Product_price',\
                         'Web_Discount','Detail_View','month_sin','month_cos']
        
            @st.cache(suppress_st_warning=True)
            def create_X():
                X= df[final_features]
                return X
            
            X=create_X()
            preprocess = pickle.load(open("FINALMODEL_preprocess_top9features.pickle","rb"))
            X_transf= preprocess.transform(X)
            
            # PREDICTIONS - modelo calibrado
            
            @st.cache(suppress_st_warning=True)
            def results_calibrated():
                model_calibrated = pickle.load(open("FINALCLF_calibrated.pickle","rb"))
                predictions = pd.Series(model_calibrated.predict(X_transf))
                
                results_calibrated=df[final_features]
                results_calibrated=pd.concat([results_calibrated,predictions],axis=1)
                results_calibrated.rename(columns={0:'CONVERSION'},inplace=True)
                return results_calibrated
            
            results_calibrated=results_calibrated()

            ## OUTPUT
            st.subheader('Número de observaciones totales:')
            st.markdown(results_calibrated['CONVERSION'].count())
            st.subheader('Número de conversiones esperadas:')
            st.markdown(results_calibrated['CONVERSION'].sum())
            st.subheader('Ratio de conversión:')
            conversion_rate=results_calibrated['CONVERSION'].sum()/results_calibrated['CONVERSION'].count()
            st.markdown(conversion_rate)
            
            discount=st.slider('Seleccione un porcentaje de descuento adicional para estimular a los indecisos:',0,50,5)
            discount=int(discount)/100
            submit=st.button('Aplicar descuento a los indecisos')
            
            if submit:
                X_class0 = results_calibrated[results_calibrated['CONVERSION']==0].drop(['CONVERSION'], axis=1)
                X_class0['Web_Discount']=X_class0['Web_Discount'].apply(lambda web_disc: web_disc+discount)
                X_class0_transf= preprocess.transform(X_class0)
                model_calibrated = pickle.load(open("FINALCLF_calibrated.pickle","rb"))
                predictions_class0 = pd.Series(model_calibrated.predict(X_class0_transf))
            
                st.subheader('Número de conversiones adicionales tras aplicar el descuento a los indecisos:')
                st.markdown(predictions_class0.sum())
                st.subheader('Incremento en porcentaje:')
                variation=np.round(predictions_class0.sum()/results_calibrated['CONVERSION'].sum()*100,2)
                st.markdown(variation)
                st.markdown('%')
                st.subheader('Ratio de conversión tras aplicar el descuento a los indecisos:')
                conversion_rate_after_discount=(results_calibrated['CONVERSION'].sum()+predictions_class0.sum())/results_calibrated['CONVERSION'].count()
                st.markdown(conversion_rate_after_discount)
                st.markdown('*Pase a la siguiente página (¿Cómo funciona?) para ver cómo se estiman \
                cada una de las conversiones >*')
            
        except NameError:
            st.error('Una vez cargado el fichero se mostrará el número total de conversiones estimadas.')
            
if choice=='¿Cómo funciona?':
        
        # DATA:
        
        @st.cache(suppress_st_warning=True)
        def load_cities():
            with open(data_root+'cities.csv',newline='') as file:
                reader = csv.reader(file)
                cities = list(reader)
            file.closed
            return cities
        
        @st.cache(suppress_st_warning=True)
        def load_prod_info():
            prod_datafile=os.path.join(data_root,'prod_info.csv')
            prod_info=pd.read_csv(prod_datafile)
            prod_info['ProductID']=prod_info['ProductID'].astype(str)
            return prod_info
        
        # INPUT:
        
        st.title('¿Cómo funciona?')
        st.subheader('Introduciendo los siguientes datos, el modelo preentrenado de Machine Learning es capaz\
        de estimar la probabilidad de conversión y si el usuario en cuestión añadirá o no dicho producto \
        al carrito (probabilidad > 0.5):')
        st.markdown('Siga las instrucciones:')
        
        productSKU=st.text_input('Introduzca la referencia (SKU) del producto para el que desea \
        realizar la predicción (Ejemplo: 1391):')
        prod_info=load_prod_info()
        try:
            product_name=prod_info[prod_info['ProductID']==productSKU]['Nombre'].values[0]
        except IndexError:
            st.error('Indique una referencia de producto válida.')
        try:
            st.markdown(product_name)
        except NameError:
            st.error('Una vez introducida la referencia, se mostrará el nombre del producto.')
        
        try:
            product_price=prod_info[prod_info['ProductID']==productSKU]['PVP'].values[0]
        except IndexError:
            st.error('Una vez introducida la referencia, se mostrará el precio del producto.')
        try:
            st.markdown(product_price)
        except NameError:
            st.error('Confirme que el precio en pantalla es válido.')
        st.markdown('EUROS')
        agree=st.checkbox('Marque la casilla para indicar que el precio mostrado es correcto.')
        if agree==True:
            price=product_price
        else:
            try:
                price=st.text_input('Si no, introduzca directamente el PVP del producto:')
                price=float(price.replace(',','.'))
            except ValueError:
                st.error('Este campo sólo acepta números.')
        
        url=st.text_input('Copie y pegue la URL específica que desea evaluar y en la que aparece el producto:')
        https="https://"
        www='www.'
        domain='galileo61.com'
        url = url.replace(https,"")
        url = url.replace(www,"")
        url = url.replace(domain,"")
        
        landing=st.text_input('Copie y pegue la URL de la Landing Page del usuario en cuestión:')
        landing = landing.replace(https,"")
        landing = landing.replace(www,"")
        landing = landing.replace(domain,"")
        
        cities=load_cities()
        city=st.selectbox('Seleccione la ciudad del usuario en cuestión:',cities[0])
        
        date=st.date_input('Seleccione la fecha para la que desea realizar la predicción:', datetime.date(2021,1,1))
        
        discount=st.slider('Seleccion el porcentaje de descuento disponible en la web para esa fecha:',0,50,15)
        discount=int(discount)/100
        
        submit=st.button('Confirmar valores')
            
        if submit:
            
            # DF    
            dict_keys= ['product_name','ga:productSKU','ga:city','ga:pagePath',\
                        'ga:landingPagePath','Product_price','Web_Discount','date']
            try:
                dict_values=[product_name,productSKU,city,url,landing,price,discount,date]
                dict_={}
                for i,column in enumerate(dict_keys):
                        dict_[column]=dict_values[i]
                df=pd.DataFrame(dict_, index=[0])      
            
                aditional_discount=[0,0.03,0.05,0.08,0.1,0.15]
                for i,value in enumerate(aditional_discount):
                    df.at[i, 'Web_Discount'] = discount+value
                df.fillna(method='ffill',inplace=True)
                
                # Feature Engineering
                df['ga:pagePath'] = df['ga:pagePath'].apply(lambda x: x[:x.find("?pag")] if "?pag" in x else x)
                df['Detail_View']=df['ga:pagePath'].apply(lambda url: 1 if url[-5:]=='.html' else 0)
                df['ga:landingPagePath'] = df['ga:landingPagePath'].apply(lambda x: x[:x.find("?pag")] if "?pag" in x else x)
                
                df['date']=pd.to_datetime(df['date'],format='%Y-%m-%d')
                df['dateTime_month']=df['date'].dt.month
                df['month_sin']=np.sin((df.dateTime_month-1)*(2.*np.pi/12))
                df['month_cos']=np.cos((df.dateTime_month-1)*(2.*np.pi/12))
                
                # PREPROCESS
        
                ## Custom Transformers
        
                class SelectColumnTransformer(BaseEstimator,TransformerMixin):
                    def fit(self,X,y=None):
                        return self
                    def transform (self,X,y=None):
                        return X
                
                def engineer_feature(columns, X):
                    df = pd.DataFrame(X, columns=columns)
                    df["Product_price"] = df["Product_price"]*(1-df["Web_Discount"])
                    return df
                
                ## Transformation
                final_features= ['ga:productSKU','ga:city','ga:pagePath','ga:landingPagePath','Product_price',\
                                 'Web_Discount','Detail_View','month_sin','month_cos']
                X= df[final_features]
                preprocess = pickle.load(open("FINALMODEL_preprocess_top9features.pickle","rb"))
                X_transf= preprocess.transform(X)
                
                # PREDICTIONS
                
                # modelo no calibrado
                model_uncalibrated = pickle.load(open("FINALCLF_uncalibrated.pickle","rb"))
                probabilities = pd.Series(np.round(model_uncalibrated.predict_proba(X_transf)[:,1],2))
                predictions = pd.Series(model_uncalibrated.predict(X_transf))
                
                results_uncalibrated=df[dict_keys]
                results_uncalibrated['date']=results_uncalibrated['date'].astype(str).str[:10]
                results_uncalibrated=pd.concat([results_uncalibrated,probabilities,predictions],axis=1)
                results_uncalibrated.rename(columns={0:'PROBABILIDAD',1:'CONVERSION'},inplace=True)
                
                x=results_uncalibrated['Web_Discount']
                y=results_uncalibrated['PROBABILIDAD']
                
                fig_uncalibrated=plt.figure(figsize=(10,8))
                ax=fig_uncalibrated.add_axes([0.5,0.2,0.6,0.5])
                l1= ax.plot(x,y, color="#F63366")
                l2=ax.axhline(y=0.5, color='tab:gray', linestyle='-')
                ax.set_ylim(bottom=0,top=1)
                ax.set_xticks(x)
                ax.set_xlabel('Porcentaje de Descuento Total ( web + adicional )')
                ax.set_ylabel('Probabilidad de convertir')
                ax.set_title('Probabilidad de convertir en base al descuento')
                
                # shap interpretation
                def st_shap(plot, height=None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height=height)
                
                X_transf=pd.DataFrame(X_transf,columns=final_features)
                explainer=shap.TreeExplainer(model_uncalibrated)
                chosen_instance = X_transf.loc[[0]]
                shap_values = explainer.shap_values(chosen_instance)
                shap.initjs()
        
                # modelo calibrado
                model_calibrated = pickle.load(open("FINALCLF_calibrated.pickle","rb"))
                probabilities = pd.Series(np.round(model_calibrated.predict_proba(X_transf)[:,1],2))
                predictions = pd.Series(model_calibrated.predict(X_transf))
                
                results_calibrated=df[dict_keys]
                results_calibrated['date']=results_calibrated['date'].astype(str).str[:10]
                results_calibrated=pd.concat([results_calibrated,probabilities,predictions],axis=1)
                results_calibrated.rename(columns={0:'PROBABILIDAD',1:'CONVERSION'},inplace=True)
                
                x=results_calibrated['Web_Discount']
                y=results_calibrated['PROBABILIDAD']
                
                fig_calibrated=plt.figure(figsize=(10,8))
                ax=fig_calibrated.add_axes([0.5,0.2,0.6,0.5])
                l1= ax.plot(x,y,color="#F63366")
                l2=ax.axhline(y=0.5, color='tab:gray', linestyle='-')
                ax.set_ylim(bottom=0,top=1)
                ax.set_xticks(x)
                ax.set_xlabel('Porcentaje de Descuento Total ( web + adicional )')
                ax.set_ylabel('Probabilidad de convertir')
                ax.set_title('Probabilidad de convertir en base al descuento')
                
                # OUTPUT
                st.title('Predicciones más optimistas:')
                st.subheader('Probabilidad de convertir sin descuento adicional:')
                st.markdown('En la siguiente visualización, se muestran los principales atributos \
                que contribuyen **positiva** (en rojo) y **negativamente** (en azul) a la probabilidad \
                de conversión en este caso específico:')
                st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], chosen_instance))
                st.subheader('Probabilidad de convertir para distintos tipos de descuento:')
                st.markdown('Así evolucionan las probabilidades, si atendemos a distintos tipos de \
                descuento adicionales:')
                st.table(results_uncalibrated)
                st.pyplot(fig_uncalibrated)
                st.title('Predicciones menos optimistas:')
                st.table(results_calibrated)
                st.pyplot(fig_calibrated)
                
            except NameError:
                st.error('Debe completar todos los campos del formulario.')
            
