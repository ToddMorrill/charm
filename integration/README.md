# Creating a Test Environment
## Set up CCU Sandbox
```
# run the following command to obtain your IP address and add it to ./sandbox/ccu-config.yaml
# NOTE: you may need to use your local ipv4 address instead (e.g. 192.168.0.125)
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
# to limit GPUs use: --gpus '"device=1,2"'
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
# to limit GPUs use: --gpus '"device=1,2"'
MODEL_SERVICE=columbia-communication-change-backend
docker run --gpus all -it --rm -p 8000:8000 --publish-all --name ${MODEL_SERVICE} columbia-communication-change-backend

# inject messages
python script.py --jsonl ./transcripts/text_M01000FLX.jsonl --fast 0.01

# you should now see that your frontend is hitting your backend with API calls!
```

## Authenticate to SRI Docker Repository (Artifactory)
```
docker login cirano-docker.cse.sri.com
```

## Update your container's yaml file and validate its formatting
```
python yamlcheck.py columbia-communication-change.yaml
```

## Pushing your frontend container to SRI's Artifactory
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

## Push your backend container to AWSs's container registry
```
<!-- # authenticate to AWS ECR
pip install boto3
# https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/sts/sts_temporary_credentials/assume_role_mfa.py

https://sts.amazonaws.com/
?Version=2011-06-15
&Action=AssumeRole
&RoleSessionName=integration-session
&RoleArn=arn:aws:iam::753712517198:role/ECR-Read,Write&
&DurationSeconds=1800
&ExternalId=123ABC
&AUTHPARAMS -->

# install awscli, if needed
# if on linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# configure aws with access key credentials
ROOTKEY=~/.ssh/yh_rootkey.csv
ROOTKEY=~/.ssh/user1_accessKeys.csv
# silly Amazon hasn't added the User Name column..
python aws_username.py --key-filepath ${ROOTKEY}
aws configure import --csv file://${ROOTKEY}

# Docker Auth using awscli (install awscli, if needed)
aws ecr get-login-password --region us-east-1 --profile yh3228 | docker login --username AWS --password-stdin 753712517198.dkr.ecr.us-east-1.amazonaws.com

# build your backend container
CONTAINER_NAME=columbia-communication-change-backend
docker build -t ${CONTAINER_NAME}:0.2 -t ${CONTAINER_NAME}:latest -f Dockerfile .

# tag this container
docker tag ${CONTAINER_NAME}:latest 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest

# push to AWS
docker push 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest
```

## Repeat this for all containers
```
# emotions
# these are dummy tags on the columbia-communication-change-backend container
CONTAINER_NAME=columbia-emotions-backend
# tag this container
docker tag columbia-communication-change-backend:latest 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest

# push to AWS
docker push 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest

# norms
CONTAINER_NAME=columbia-norms-backend
# tag this container
docker tag columbia-communication-change-backend:latest 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest

# push to AWS
docker push 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest

# vision
CONTAINER_NAME=columbia-vision-backend
# tag this container
docker tag columbia-communication-change-backend:latest 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest

# push to AWS
docker push 753712517198.dkr.ecr.us-east-1.amazonaws.com/integration/${CONTAINER_NAME}:latest
```

## Running everything together locally
```
# in the main integration folder run
docker-compose up

# you can test all APIs with query.py in the backend directory
python query.py
```

## Running on AWS
```
# create a g4dn:12xlarge instance under security group 17

# Install docker-compose
sudo apt install docker-compose

# copy the docker-compose.yaml file
nano docker-compose.yml
# paste the contents

# authenticate to ECR
# install aws cli
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# copy access key credentials
ROOTKEY=~/.ssh/yh_rootkey.csv
nano ${ROOTKEY}
# paste contents from yh_rootkey.csv

# configure AWS CLI
aws configure import --csv file://${ROOTKEY}

# Docker Auth using awscli (install awscli, if needed)
aws ecr get-login-password --region us-east-1 --profile yh3228 | docker login --username AWS --password-stdin 753712517198.dkr.ecr.us-east-1.amazonaws.com

# run all images (-d to run in background)
docker compose up -d
```