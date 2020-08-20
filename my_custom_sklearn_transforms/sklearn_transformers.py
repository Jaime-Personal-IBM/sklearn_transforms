from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class RellenarFaltantes(BaseEstimator, TransformerMixin):
    def __init__(self, cols_mediana, cols_ceros):
        self.cols_mediana = cols_mediana
        self.cols_ceros = cols_ceros

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        
        si01 = SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='median',  # la estrategia elegida es cambiar el valor faltante por una constante
          #  fill_value=0,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )
        si02 = SimpleImputer(
            missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
            strategy='constant',  # la estrategia elegida es cambiar el valor faltante por una constante
            fill_value=0,  # la constante que se usará para completar los valores faltantes es un int64 = 0
            verbose=0,
            copy=True
        )
        
        
        data[self.cols_mediana] = si01.fit_transform(data[self.cols_mediana])
        data[self.cols_ceros] = si02.fit_transform(data[self.cols_ceros])
        return data
