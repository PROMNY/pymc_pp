init:
	pip3 install --user -r requirements.txt

install:
	pip3 install --user .

test_short:
	coverage run -m --source=./pymc pytest -m "not long"
	coverage report -m

test:
	coverage run -m --source=./pymc pytest
	coverage report -m