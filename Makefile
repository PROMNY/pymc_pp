init:
	pip install -r requirements.txt

install:
	pip install --user .

test:
	nosetests tests
