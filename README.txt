#Build docker image from Dockerfile
docker build -t dic-assignment .

# Run docker container locally
docker run --name dic -v dic-images:/app/images -d -p 5000:5000 dic-assignment

#run inference on single image files from a path
curl http://localhost:5000/api/detect -d "input=./images/filename.jpg"

#to save annotated images, add a "flag" by adding any character to "output"
curl http://localhost:5000/api/detect -d "input=./images/filename.jpg&output=1"

##Cluster

#docker build on cluster
docker build -t dic-assignmentg02 .

#docker run on cluster
docker run --name dic-assignmentg02 -v dic02-images:/app/images -d -p 5002:5000 dic-assignmentg02:latest

#curl cluster
curl http://s25.lbd.hpc.tuwien.ac.at:5002/api/detect -d "input=./testimages/000000000019.jpg&output=1"