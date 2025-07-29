# Gestor de Cartera Financiera

Este proyecto proporciona una aplicación web desarrollada en Python con Streamlit que permite a múltiples usuarios gestionar sus carteras de inversión de forma sencilla y segura. Incluye autenticación con contraseñas cifradas, almacenamiento de carteras por usuario, carga y descarga de archivos y análisis en tiempo real de las posiciones.

## Estructura del proyecto

```
financial_portfolio_manager/
├── app.py                # Script principal de Streamlit
├── auth.py               # Módulo de autenticación (registro y login)
├── portfolio_utils.py    # Funciones auxiliares para trabajar con carteras
├── requirements.txt      # Dependencias para ejecutar la aplicación
├── README.md             # Esta documentación
└── user_data/            # Se crea automáticamente para credenciales y carteras
```

### Principales módulos

* **app.py**: contiene la interfaz de usuario en Streamlit. Gestiona la autenticación, navegación y visualización de métricas, además de permitir subir y crear carteras.
* **auth.py**: implementa el almacenamiento y verificación segura de credenciales mediante un algoritmo de hashing con sal y PBKDF2.
* **portfolio_utils.py**: facilita el guardado/carga de carteras, la obtención de precios actuales con `yfinance`, el cálculo de métricas (valor total, beneficio/pérdida, peso, RSI, volatilidad), gráficos de distribución y sugerencias de diversificación.

## Cómo ejecutar localmente

1. Clona este repositorio y sitúate en la carpeta `financial_portfolio_manager`.
2. Instala las dependencias en un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Ejecuta la aplicación con Streamlit:

```bash
streamlit run app.py
```

La interfaz se abrirá en tu navegador por defecto. Desde ahí podrás registrarte como nuevo usuario, iniciar sesión, cargar o crear tu cartera y explorar los análisis.

## Despliegue en Streamlit Cloud

Para desplegar la aplicación en [Streamlit Cloud](https://streamlit.io/cloud) y gestionar actualizaciones desde GitHub, sigue estos pasos:

1. **Sube el repositorio a GitHub**. Asegúrate de incluir todos los archivos del proyecto (`app.py`, `auth.py`, `portfolio_utils.py`, `requirements.txt`, etc.).

2. **Crea una cuenta en Streamlit Cloud** (gratuita) e inicia sesión.

3. En tu panel de Streamlit Cloud, haz clic en *New app* y selecciona el repositorio de GitHub donde se encuentra tu proyecto. Indica la rama (por ejemplo, `main`) y el archivo principal (`financial_portfolio_manager/app.py`).

4. Streamlit Cloud instalará automáticamente las dependencias especificadas en `requirements.txt` y desplegará la aplicación. Después de unos instantes tendrás una URL pública que podrás compartir.

5. **Actualizaciones**: cuando modifiques el código en GitHub (por ejemplo, añadiendo nuevas funciones o corrigiendo errores), Streamlit Cloud detectará los cambios y actualizará automáticamente la aplicación. Procura hacer *commits* claros y subirlos a la rama seleccionada en el despliegue para que los cambios se apliquen.

## Consideraciones de seguridad

* Las contraseñas se almacenan siempre cifradas mediante PBKDF2 con sal aleatoria y 100 000 iteraciones.
* Cada usuario tiene un espacio separado en `user_data/` para sus credenciales y carteras.
* Aun así, no se recomienda usar este proyecto en producción sin revisar aspectos como autenticación de múltiples factores, cifrado en tránsito (HTTPS), bases de datos más robustas y un control de acceso más granular.

## Posibles mejoras

* Integrar APIs de datos profesionales para obtener cotizaciones en tiempo real de mayor calidad.
* Incluir estrategias de optimización de carteras (varianza mínima, frontera eficiente, etc.).
* Añadir notificaciones o alertas cuando se superen determinados umbrales de beneficio/pérdida.
* Soportar más clases de activos y permitir cargar histórico para evaluar rendimiento a largo plazo.
