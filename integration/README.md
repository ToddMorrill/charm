# Creating a Test Environment
## Set up CCU Sandbox
```
# run the following command to obtain your IP address and add it to ./sandbox/ccu-config.yaml
python -c "import socket; print(f'IP is {socket.gethostbyname(socket.gethostname())}')"

# export the path to this config file
# or add it to your ~/.bashrc (or equivalent) profile
export CCU_SANDBOX=/home/iron-man/Documents/charm/integration/sandbox
```

## Create test jsonl data
```
# create a sample text, audio, and video jsonl file
python create_jsonl.py --data-dir ~/Documents/data/charm/raw/LDC2022E11_CCU_TA1_Mandarin_Chinese_Development_Source_Data_R1

# validate the format
python jsonlcheck.py ./transcripts/audio_M01000537.jsonl
python jsonlcheck.py ./transcripts/text_M01000FLX.jsonl
python jsonlcheck.py ./transcripts/video_M010009A4.jsonl
```

## Publishing and subscribing to the mock data stream
```
# start ccuhub
python ccuhub.py

# start the logger
python -u logger.py

# start the changepoint detection system
python -u main.py

# inject messages in the CCU environment
python script.py --jsonl ./transcripts/audio_M01000537.jsonl --fast 0.01
python script.py --jsonl ./transcripts/text_M01000FLX.jsonl --fast 0.01
python script.py --jsonl ./transcripts/video_M010009A4.jsonl --fast 0.01
```

## Running the backend
```
# in the backend directory
uvicorn main:app --reload

# run query.py to test your model
python query.py
```

## Generate output JSONL files
```
TODO
```

## Running the frontend in Docker
```
# build the front end docker container with
CONTAINER_NAME=columbia-communication-change
# if using the netrc file and on linux, include buildkit env var
# more detail here: https://docs.docker.com/build/buildkit/#getting-started
DOCKER_BUILDKIT=1 docker build -t ${CONTAINER_NAME}:0.2 -t ${CONTAINER_NAME}:latest -f Dockerfile --secret id=netrc,src=../netrc .

# run your container
docker run -it --rm --publish-all --volume $CCU_SANDBOX:/sandbox ${CONTAINER_NAME}

# with ccuhub.py and logger.py running, inject the jsonl messages again
# confirm that your container is processing the messages
```

## Running the backend in Docker
```
CONTAINER_NAME=columbia-communication-change-backend
docker build -t ${CONTAINER_NAME}:0.2 -t ${CONTAINER_NAME}:latest -f Dockerfile .

# assuming you've installed nvidia-container-toolkit
# run your model in the container
docker run --gpus all -it --rm -p 8000:8000 --publish-all ${CONTAINER_NAME}

# test that you can query the model
python query.py
```

## Put it all together
```
# with ccuhub.py and logger.py running

# unfortunately inter-container communication is non-trivial without docker-compose
# so we'll just run our frontend directly on the host instead of in Docker 
<!-- MODEL_SERVICE=columbia-communication-change-backend
docker run -it --rm --publish-all --volume $CCU_SANDBOX:/sandbox \
    --env MODEL_SERVICE=${MODEL_SERVICE} \
    columbia-communication-change  -->

# in the frontend directory run
export MODEL_SERVICE=127.0.0.1
python -u main.py

# run backend and give it a name (name is optional)
MODEL_SERVICE=columbia-communication-change-backend
docker run --gpus all -it --rm -p 8000:8000 --publish-all --name ${MODEL_SERVICE} columbia-communication-change-backend

# inject messages
python script.py --jsonl ./transcripts/text_M01000FLX.jsonl --fast 0.01

# you should now see that your frontend is hitting your backend with API calls!
```

## Authenticate to remote Docker
```
docker login cirano-docker.cse.sri.com
```

## Update your container's yaml file and validate its formatting
```
python yamlcheck.py columbia-communication-change.yaml
```

## Pushing your container to SRI's Artifactory
```
DOCKER_BUILDKIT=1 docker build \
    -t cirano-docker.cse.sri.com/columbia-communication-change:0.2 \
    -t ${CONTAINER_NAME}:latest \
    -f Dockerfile \
    --secret id=netrc,src=../netrc \
    .
docker tag cirano-docker.cse.sri.com/columbia-communication-change:0.2 \
    cirano-docker.cse.sri.com/columbia-communication-change:latest
docker push cirano-docker.cse.sri.com/columbia-communication-change:0.2
docker push cirano-docker.cse.sri.com/columbia-communication-change:latest
```

## Pushing a zipped directory of your container yaml file and input/output files
```
# make a staging directory with the same name as the container
mkdir -p columbia-communication-change

# copy files to staging directory
# TODO: add output files
cp columbia-communication-change.yaml ./transcripts/audio_M01000537.jsonl ./transcripts/video_M010009A4.jsonl ./transcripts/text_M01000FLX.jsonl ./columbia-communication-change

# zip directory
zip -r columbia-communication-change.zip columbia-communication-change

# push this zip file to artifactory
curl -H 'X-JFrog-Art-Api:'"${ARTIFACTORY_APIKEY}" -T columbia-communication-change.zip \
"https://artifactory.sri.com/artifactory/cirano-local/columbia-communication-change/columbia-communication-change.zip"
```

# TODO:
- produce output jsonl
- create container yaml file

