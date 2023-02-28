# for all files in the ./columbia-communication-change directory, push them to the server
for file in ./columbia-communication-change/*
do  
    echo "Pushing $file to artifactory\n"
    curl -H 'X-JFrog-Art-Api:'"${ARTIFACTORY_APIKEY}" -T $file \
"https://artifactory.sri.com/artifactory/cirano-local/columbia-communication-change/$(basename $file)"
    echo "\n"
done