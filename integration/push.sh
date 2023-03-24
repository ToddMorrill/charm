# for all files in the ./columbia-communication-change directory, push them to the server
CONTAINER_NAME=columbia-communication-change
ARTIFACTORY_APIKEY=AKCp8nz2Ge3XbpRFeyZQhtgh3MaTpDEiX8fVpN3NAoDxAA8EBj7xDyVQDYiLYF81sgbJFG5xj
for file in ./${CONTAINER_NAME}/*
do  
    echo "Pushing $file to artifactory\n"
    curl -H 'X-JFrog-Art-Api:'"${ARTIFACTORY_APIKEY}" -T $file \
"https://artifactory.sri.com/artifactory/cirano-local/${CONTAINER_NAME}/$(basename $file)"
    echo "\n"
done