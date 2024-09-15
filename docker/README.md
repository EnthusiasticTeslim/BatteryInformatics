# Docker implementation

This folder allows using Docker for easy setup and evaluation.

## Prerequisites

- [Docker](https://docs.docker.com/engine/install/) installed on your system

## Building the Project

To build the Docker image for this project, navigate to the project's parent directory (e.g. `BatteryInformatics`) and run the following command:

```
docker build -f docker/descriptor/Dockerfile -t descriptor .
```

This command builds a Docker image named `descriptor` or `graph` using the Dockerfile located at `docker/descriptor/Dockerfile` and `docker/graph/Dockerfile`, respectively.

## Running the Project

To run the project, use the following Docker command:

Descriptor-based Model

```
docker run -it --rm --name descriptor descriptor
```

Graph Neural Network Model

```
docker run -it --rm --name graph graph
```

This command will:
- Start a new container from the image (`descriptor` or `graph`)
- Name the container `descriptor` (or `graph`)
- Run the container in interactive mode (`-it`)
- Automatically remove the container when it exits (`--rm`)