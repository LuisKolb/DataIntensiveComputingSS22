# Object Detection using a HTTP endpoint with Flask, PIL, TensorFlow, TfHub

## Running locally
Build docker image from Dockerfile  
`docker build -t dic-assignment .`

Run docker container locally  
`docker run --mount -v testimages:/images --name dic -v dic-images:/app/images -d -p 5000:5000 dic-assignment`

Run inference on single image files from a path  
`curl http://localhost:5000/api/detect -d "input=./images/filename.jpg"`

To save annotated images, add a "flag" by adding any character to "output"  
`curl http://localhost:5000/api/detect -d "input=./images/filename.jpg&output=1"`

---  

## Deploying on the LBD cluster
Transfer the cloned repository to the cluster  
`scp -r ./src group02@s25.lbd.hpc.tuwien.ac.at:/home/group02`

Docker build on cluster  
`docker build -t dic_assignmentg02 .`

Docker run on cluster  
`docker run --name dic_assignmentg02 -v ~/dic02-images:/app/images -d -p 5002:5000 dic_assignmentg02:latest`

curl cluster externally  
`curl http://s25.lbd.hpc.tuwien.ac.at:5002/api/detect -d "input=/app/images/000000000019.jpg&output=1`

Get container bash  
`docker exec -it dic_assignmentg02 /bin/bash`

Execute sender script to spread the workload across multiple nodes  
In this example, we run the `sender.py` script inside the docker container, hence localhost url  
`python sender.py /app/images/ --output --nodes 127.0.0.1:5000`

See `python sender.py -h` for more info on the script  
