

# A Dockerfile is a text document that contains all the commands
# a user could call on the command line to assemble an image.

FROM python:3.9.4-buster

# Our Debian with python is now installed.

RUN mkdir build

# We create folder named build for our stuff.

WORKDIR /build

# Now we just want to our WORKDIR to be /build

COPY . .

# FROM [path to files from the folder we run docker run]
# TO [current WORKDIR]
# We copy our files (files from .dockerignore are ignored)
# to the WORKDIR

RUN pip install --no-cache-dir -r requirements.txt

# OK, now we pip install our requirements

EXPOSE 8000

# Instruction informs Docker that the container listens on port 8000

WORKDIR /build/app

# Now we just want to our WORKDIR to be /build/app for simplicity

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# This command runs our uvicorn server

"""
# build docker image
docker build . -t apiserver
#run docker image
#acces at http://localhost:8000
docker run --rm -it  -p 8000:8000/tcp apiserver:latest
"""