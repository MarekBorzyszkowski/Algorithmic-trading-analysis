#!/bin/bash
if [ ! -d venv ]; then
	echo "Start of creating venv"
	python3 -m venv venv
	echo "venv created\nStart of installing python packages"
	source venv/bin/activate
	pip install -r requirements.txt
	deactivate
	echo "End of installing python packages"
else
	echo "venv already installed!"
fi