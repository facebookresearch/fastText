FROM gcr.io/tink-containers/debian:buster as builder
RUN apt-get update && apt-get install -y cmake g++
COPY . /project
WORKDIR /project
# Run make clean to make sure that we don't copy results from the host
RUN make clean && make
FROM gcr.io/distroless/base
COPY --from=builder /project/fasttext /