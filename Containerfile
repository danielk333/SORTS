# Use the official python image as the base image
FROM python:3.11.9

# Dependencies
RUN apt update

# Install package
COPY . /sorts
WORKDIR /sorts
RUN pip install --upgrade pip && \
    pip install .
