init:
	pip3 install --user -r requirements.txt

install:
	pip3 install --user .

test:
	coverage run -m --source=./pymc pytest 
	coverage report -m
