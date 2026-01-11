FROM python 3.13.7   
### from Docker hub, takes a base linux image and on the top of that python 3.13.7 
COPY . /app                 
### Copy all the files in our folder within that base image, . means from current location /app means to teh app we have created  
WORKDIR /app         
### is the work directory to which we have copied our current location 
RUN pip install -r requirements.txt
EXPOSE  $PORT   
                # when inside our docker image when docker image is run as a container in order to 
                # access to application inside teh container so we have to expose some port that woudl 
                # enable to access teh entire url and we will access application from that portal 
                # so we expose a port within that docker container and that port will be called as a 
                # placeholder which is called as dollar port because this value when we deploy into the 
                # cloud or server 
CMD gunicorn --workers=4  --bind 0.0.0.0:$PORT app:app
# 0.0.0.0 is the local ip address that woudl be binded with the $PORT