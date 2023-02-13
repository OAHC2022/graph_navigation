#!/bin/bash
echo $(dirname $(realpath $0))
cd $(dirname $(realpath $0))
./bin/navigation