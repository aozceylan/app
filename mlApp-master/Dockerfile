FROM python:3.9-slim
#python image als basis

# arbeitsverzeichnis im Container setzen
WORKDIR /app

#kopieren des aktuellen verzeichnisses in /app
ADD . /app

# kopiert aus lokalem folder 'datasets' in '/app/datensätze' im container
COPY ./datasets /app/data


# abhängigkeiten installieren
RUN pip install --trusted-host pypi.python.org --no-cache-dir -r requirements.txt


# port auf dem streamlit lauscht
EXPOSE 8501

# RESTART ALWAYS BEFEHL EINFÜGEN

CMD ["streamlit", "run", "app.py"]
