#!/bin/bash

# First `uv sync --extra docs` and `uv run mkdocs build`
# Then run this script.

rm -rf build;
mkdir build;
cd build;
git init -b gh-pages;
git remote add origin git@github.com:WiseLabCMU/xwr.git;
cd ..;
cp -r site/* build/;
cd build;
touch .nojekyll;
git add --all;
git commit -m "Update gh-pages";
git push -f origin gh-pages;
