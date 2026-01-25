#!/bin/bash

DIR="$1"

if [ -z "$DIR" ]; then
    echo "use $0 <directory_name>"
    exit 1
fi

if [ ! -d "$DIR" ]; then
    mkdir "$DIR"
    echo "new dir $DIR"
fi

mv *.mp4 *.png *.pdf *.vti "$DIR" 2>/dev/null

echo "done"
