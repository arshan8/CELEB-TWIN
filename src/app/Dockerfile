
#goal: i wanna make an image that will make my project run, this docker file is recipe for that image

#pul a base image which gives all required tool and libraries
FROM python:3.11-slim

#create a folder where the app code will be stored
WORKDIR /app

#copy the source code from your Host machine to your container
COPY src/app/requirements.txt /app/requirements.txt

#compile the application code
RUN pip install -r requirements.txt

COPY src/app /app
COPY src/vector_twin /app/vector_twin

EXPOSE 8501

#run the application #cmd can be overwritten
CMD ["streamlit", "run", "main.py"]

#ENTRYPOINT: bada bhai of CMD which cannot be overwritten