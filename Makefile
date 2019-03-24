init:
	pip3 install --user -r requirements.txt

install:
	pip3 install --user .

test:
	py.test-3
